#!/usr/bin/python3

import numpy as np
import gym, torch
import Net, PolicyLearner, DataHandler

def main():
    env = gym.make('CartPole-v0')
    net = Net.Net()
    policy = PolicyLearner.PolicyLearner(net)
    dataHandler = DataHandler.DataHandler(policy, env)

    for i in range(1000):
        continueFlag = dataHandler.render(episodes=3)
        if not continueFlag:
            return

        reward = dataHandler.generate(episodes=2000)
        dataHandler.train(batchSize=32)
        dataHandler.reset()
        print("reward:", reward)


if __name__ == "__main__":
    main()
