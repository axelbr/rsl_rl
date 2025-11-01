# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=wrong-import-position
"""Train a PPO agent using RSL-RL for the specified environment."""

import os

from tensordict import TensorDict

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import functools
import json
import numpy as np
import os
import torch
from collections import deque
from datetime import datetime

import jax
import mediapy as media
import mujoco
import mujoco_playground
import wandb
from absl import app, flags, logging
from ml_collections import config_dict
from mujoco_playground import registry, wrapper_torch
from mujoco_playground.config import locomotion_params, manipulation_params

from rsl_rl.runners import OnPolicyRunner

try:
    from rsl_rl.env import VecEnv  # pytype: disable=import-error
except ImportError:
    VecEnv = object
try:
    import torch  # pytype: disable=import-error
except ImportError:
    torch = None

from mujoco_playground._src import wrapper

# Suppress logs if you want
logging.set_verbosity(logging.WARNING)

# Define flags similar to the JAX script
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "BerkeleyHumanoidJoystickFlatTerrain",
    (f"Name of the environment. One of: {', '.join(mujoco_playground.registry.ALL_ENVS)}"),
)
_LOAD_RUN_NAME = flags.DEFINE_string("load_run_name", None, "Run name to load from (for checkpoint restoration).")
_CHECKPOINT_NUM = flags.DEFINE_integer("checkpoint_num", -1, "Checkpoint number to load from.")
_PLAY_ONLY = flags.DEFINE_boolean("play_only", False, "If true, only play with the model and do not train.")
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode).",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name.")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 4096, "Number of parallel envs.")
_DEVICE = flags.DEFINE_string("device", "cuda:0", "Device for training.")
_MULTI_GPU = flags.DEFINE_boolean("multi_gpu", False, "If true, use multi-GPU training (distributed).")
_CAMERA = flags.DEFINE_string("camera", None, "Camera name to use for rendering.")


def _jax_to_torch(tensor):
    import torch.utils.dlpack as tpack  # pytype: disable=import-error # pylint: disable=import-outside-toplevel

    tensor = tpack.from_dlpack(tensor)
    return tensor


def _torch_to_jax(tensor):
    from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel

    tensor = from_dlpack(tensor)
    return tensor


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except Exception as exc:
        raise ValueError("No runs in this directory: " + root) from exc
    if load_run == -1 or load_run == "-1":
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: m.zfill(15))
        model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"

    load_path = os.path.join(load_run, model)
    return load_path


class RSLRLBraxWrapper(VecEnv):
    """Wrapper for Brax environments that interop with torch."""

    def __init__(
        self,
        env,
        num_actors: int,
        seed: int,
        episode_length: int,
        action_repeat: int,
        randomization_fn=None,
        render_callback=None,
        device_rank=None,
    ) -> None:
        import torch  # pytype: disable=import-error # pylint: disable=redefined-outer-name,unused-import,import-outside-toplevel

        self.seed = seed
        self.batch_size = num_actors
        self.num_envs = num_actors

        self.key = jax.random.PRNGKey(self.seed)

        if device_rank is not None:
            gpu_devices = jax.devices("gpu")
            self.key = jax.device_put(self.key, gpu_devices[device_rank])
            self.device = f"cuda:{device_rank}"
            print(f"Device -- {gpu_devices[device_rank]}")
            print(f"Key device -- {self.key.devices()}")

        # split key into two for reset and randomization
        key_reset, key_randomization = jax.random.split(self.key)

        self.key_reset = jax.random.split(key_reset, self.batch_size)

        if randomization_fn is not None:
            randomization_rng = jax.random.split(key_randomization, self.batch_size)
            v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
        else:
            v_randomization_fn = None

        self.env = wrapper.wrap_for_brax_training(
            env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )

        self.render_callback = render_callback

        self.asymmetric_obs = False
        obs_shape = self.env.env.unwrapped.observation_size
        print(f"obs_shape: {obs_shape}")

        if isinstance(obs_shape, dict):
            print("Asymmetric observation space")
            self.asymmetric_obs = True
            self.num_obs = obs_shape["state"]
            self.num_privileged_obs = obs_shape["privileged_state"]
        else:
            self.num_obs = obs_shape
            self.num_privileged_obs = None

        self.num_actions = self.env.env.unwrapped.action_size

        self.max_episode_length = episode_length

        # todo -- specific to leap environment
        self.success_queue = deque(maxlen=100)

        print("JITing reset and step")
        self.reset_fn = jax.jit(self.env.reset)
        self.step_fn = jax.jit(self.env.step)
        print("Done JITing reset and step")
        self.env_state = None

    def step(self, action):
        action = torch.clip(action, -1.0, 1.0)  # pytype: disable=attribute-error
        action = _torch_to_jax(action)
        self.env_state = self.step_fn(self.env_state, action)
        critic_obs = None
        if self.asymmetric_obs:
            obs = _jax_to_torch(self.env_state.obs["state"])
            critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
            obs = TensorDict({"state": obs, "privileged_state": critic_obs})
        else:
            obs = _jax_to_torch(self.env_state.obs)
            obs = TensorDict({"state": obs})
        reward = _jax_to_torch(self.env_state.reward)
        done = _jax_to_torch(self.env_state.done)
        info = self.env_state.info
        truncation = _jax_to_torch(info["truncation"])

        info_ret = {
            "time_outs": truncation,
            "observations": {"critic": critic_obs},
            "log": {},
        }

        if "last_episode_success_count" in info:
            last_episode_success_count = (
                _jax_to_torch(info["last_episode_success_count"])[done > 0]  # pylint: disable=unsubscriptable-object
                .float()
                .tolist()
            )
            if len(last_episode_success_count) > 0:
                self.success_queue.extend(last_episode_success_count)
            info_ret["log"]["last_episode_success_count"] = np.mean(self.success_queue)

        for k, v in self.env_state.metrics.items():
            if k not in info_ret["log"]:
                info_ret["log"][k] = _jax_to_torch(v).float().mean().item()

        return obs, reward, done, info_ret

    def reset(self):
        # todo add random init like in collab examples?
        self.env_state = self.reset_fn(self.key_reset)

        if self.asymmetric_obs:
            obs = _jax_to_torch(self.env_state.obs["state"])
            # critic_obs = jax_to_torch(self.env_state.obs["privileged_state"])
        else:
            obs = _jax_to_torch(self.env_state.obs)
        return obs

    def reset_with_critic_obs(self):
        self.env_state = self.reset_fn(self.key_reset)
        obs = _jax_to_torch(self.env_state.obs["state"])
        critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
        return obs, critic_obs

    def get_observations(self):
        if self.asymmetric_obs:
            obs, critic_obs = self.reset_with_critic_obs()
            return TensorDict({"state": obs, "privileged_state": critic_obs})
        else:
            obs = self.reset()
            return TensorDict({"state": obs})

    def render(self, mode="human"):  # pylint: disable=unused-argument
        if self.render_callback is not None:
            self.render_callback(self.env.env.env, self.env_state)
        else:
            raise ValueError("No render callback specified")

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info["action_space"] = self.action_space  # pytype: disable=attribute-error
        info["observation_space"] = self.observation_space  # pytype: disable=attribute-error
        return info


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    if env_name in registry.manipulation._envs:
        return manipulation_params.rsl_rl_config(env_name)
    elif env_name in registry.locomotion._envs:
        return locomotion_params.rsl_rl_config(env_name)
    else:
        raise ValueError(f"No RL config for {env_name}")


def main(argv):
    """Run training and evaluation for the specified environment using RSL-RL."""
    del argv  # unused

    # Possibly parse the device for multi-GPU
    if _MULTI_GPU.value:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_rank = local_rank
        device = f"cuda:{local_rank}"
        print(f"Using multi-GPU: local_rank={local_rank}, device={device}")
    else:
        device = _DEVICE.value
        device_rank = int(device.split(":")[-1]) if "cuda" in device else 0

    # If play-only, use fewer envs
    num_envs = 1 if _PLAY_ONLY.value else _NUM_ENVS.value

    # Load default config from registry
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    print(f"Environment config:\n{env_cfg}")

    # Generate unique experiment name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{_ENV_NAME.value}-{timestamp}"
    if _SUFFIX.value is not None:
        exp_name += f"-{_SUFFIX.value}"
    print(f"Experiment name: {exp_name}")

    # Logging directory
    logdir = os.path.abspath(os.path.join("logs", exp_name))
    os.makedirs(logdir, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # Checkpoint directory
    ckpt_path = os.path.join(logdir, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # Initialize Weights & Biases if required
    if _USE_WANDB.value and not _PLAY_ONLY.value:
        wandb.tensorboard.patch(root_logdir=logdir)
        wandb.init(project="mjxrl", name=exp_name)
        wandb.config.update(env_cfg.to_dict())
        wandb.config.update({"env_name": _ENV_NAME.value})

    # Save environment config to JSON
    with open(os.path.join(ckpt_path, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    # Domain randomization
    randomizer = registry.get_domain_randomizer(_ENV_NAME.value)

    # We'll store environment states during rendering
    render_trajectory = []

    # Callback to gather states for rendering
    def render_callback(_, state):
        render_trajectory.append(state)

    # Create the environment
    raw_env = registry.load(_ENV_NAME.value, config=env_cfg)
    brax_env = RSLRLBraxWrapper(
        raw_env,
        num_envs,
        _SEED.value,
        env_cfg.episode_length,
        1,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )

    # Build RSL-RL config
    train_cfg = get_rl_config(_ENV_NAME.value)

    obs_size = raw_env.observation_size
    if isinstance(obs_size, dict):
        train_cfg.obs_groups = {"policy": ["state"], "critic": ["privileged_state"]}
    else:
        train_cfg.obs_groups = {"policy": ["state"], "critic": ["state"]}

    # Overwrite default config with flags
    train_cfg.seed = _SEED.value
    train_cfg.run_name = exp_name
    train_cfg.resume = _LOAD_RUN_NAME.value is not None
    train_cfg.load_run = _LOAD_RUN_NAME.value if _LOAD_RUN_NAME.value else "-1"
    train_cfg.checkpoint = _CHECKPOINT_NUM.value

    train_cfg_dict = train_cfg.to_dict()
    runner = OnPolicyRunner(brax_env, train_cfg_dict, logdir, device=device)

    # If resume, load from checkpoint
    if train_cfg.resume:
        resume_path = wrapper_torch.get_load_path(
            os.path.abspath("logs"),
            load_run=train_cfg.load_run,
            checkpoint=train_cfg.checkpoint,
        )
        print(f"Loading model from checkpoint: {resume_path}")
        runner.load(resume_path)

    if not _PLAY_ONLY.value:
        # Perform training
        runner.learn(
            num_learning_iterations=train_cfg.max_iterations,
            init_at_random_ep_len=False,
        )
        print("Done training.")
        return

    # If just playing (no training)
    policy = runner.get_inference_policy(device=device)

    # Example: run a single rollout
    eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(_SEED.value)
    state = jit_reset(rng)
    rollout = [state]

    # We’ll assume your environment’s observation is in state.obs["state"].
    obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])

    for _ in range(env_cfg.episode_length):
        with torch.no_grad():
            actions = policy(obs_torch)
        # Step environment
        state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
        rollout.append(state)
        obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])
        if state.done:
            break

    # Render
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    render_every = 2
    # If your environment is wrapped multiple times, adjust as needed:
    base_env = eval_env  # or brax_env.env.env.env
    fps = 1.0 / base_env.dt / render_every
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj,
        camera=_CAMERA.value,
        height=480,
        width=640,
        scene_option=scene_option,
    )
    media.write_video("rollout.mp4", frames, fps=fps)
    print("Rollout video saved as 'rollout.mp4'.")


if __name__ == "__main__":
    app.run(main)
