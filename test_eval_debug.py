import warnings
import traceback

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

import sys
import yaml
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack
from env_sailing_env import ImprovedSailingEnv

def test():
    raw_env = ImprovedSailingEnv(render_mode="rgb_array")
    raw_env = ss.black_death_v3(raw_env)
    venv = ss.pettingzoo_env_to_vec_env_v1(raw_env)
    venv = ss.concat_vec_envs_v1(venv, 1, num_cpus=0, base_class="stable_baselines3")
    
    # MAGIC FIX HERE!
    venv.render_mode = "rgb_array"
    
    venv = VecMonitor(venv)
    print("SUCCESS: VecMonitor initialized without warning!")
test()
