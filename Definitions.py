
import random, os
import numpy as np
import torch, gym

# cpu and cuda runs are both deterministic, but cpu runs give differing results with cuda runs.
# It is not clear whether runs differ on different machines.

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
deterministic = 2 # 0 - non-deterministic, 1 - no performance-intrusive determinism, 2 - maximally deterministic (affects only cuda)
modelDir = "models/"

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


def saveModel(net, envName, iteration, reward):
    modelName = "%s_%03d_%.2f.pth" % (envName, iteration, reward)
    os.makedirs(modelDir, exist_ok=True)
    torch.save(net, modelDir + modelName)


def loadModel(envName, iteration=None, loadBest=False):
    assert (iteration is not None and loadBest == False) or (iteration is None and loadBest == True)
    files = os.listdir(modelDir)
    modelFiles = [ ModelFile(i) for i in files ]
    modelFiles = list(filter(lambda x: x.envName == envName, modelFiles))

    if iteration is not None:
        correctFiles = [ i for i in modelFiles if i.iteration == iteration ]
        if len(correctFiles) > 1: print("Warning: more than 1 match found for %s: %d" % (envName, iteration))
        correctName = modelDir + correctFiles[0].name
        return torch.load(correctName)

    if loadBest:
        modelFiles.sort(key=lambda x: x.reward)
        correctName = modelDir + modelFiles[-1].name
        return torch.load(correctName)


class ModelFile:
    def __init__(self, name):
        self.name = name
        split = name.split("_")
        self.envName = split[0]
        self.iteration = int(split[1])
        self.reward = float(split[2][:-4]) # [:-4] removes .pth extension
