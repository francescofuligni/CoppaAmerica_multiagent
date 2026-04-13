import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack
from env_sailing_env import ImprovedSailingEnv
import yaml

env = ImprovedSailingEnv(render_mode="rgb_array")
venv = ss.pettingzoo_env_to_vec_env_v1(env)
venv = ss.concat_vec_envs_v1(venv, 1, num_cpus=0, base_class="stable_baselines3")
venv = VecMonitor(venv)
venv = VecFrameStack(venv, n_stack=4)
obs = venv.reset()
print("VENV OBS SHAPE:", obs.shape)
venv.close()
