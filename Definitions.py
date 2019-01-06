
import random
import numpy as np
import torch, gym

# cpu and cuda runs are both deterministic, but cpu runs give differing results with cuda runs.
# It is not clear whether runs differ on different machines.

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
deterministic = 2 # 0 - non-deterministic, 1 - no performance-intrusive determinism, 2 - maximally deterministic (affects only cuda)

print("Device:", device, "   determinism level:", deterministic)

if deterministic >= 1:
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

if deterministic >= 2:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def makeSeededEnvironment(name):
    env = gym.make(name)
    if deterministic == 0:
        return env

    global envSeed
    try:
        envSeed += 1
    except NameError:
        envSeed = 1
    env.seed(envSeed)
    return env
