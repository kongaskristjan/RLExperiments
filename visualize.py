#!/usr/bin/python3

import torch, gym
import Definitions, PolicyLearner, DataHandler

envName = "LunarLander-v2"
iteration = None
confidenceMul = 1000

def main():
    env = gym.make(envName)
    net = Definitions.loadModel(envName, iteration=iteration, loadBest=(iteration is None))
    policy = PolicyLearner.PolicyLearner(net)
    policy.setConfidenceMul(confidenceMul)
    dataHandler = DataHandler.DataHandler(policy, env)
    dataHandler.render(episodes=int(1e9))


if __name__ == "__main__":
    main()
