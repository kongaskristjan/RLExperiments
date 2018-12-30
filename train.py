#!/usr/bin/python3

import numpy as np
import gym, torch
import PolicyLearner, DataHandler

def main():
    env = gym.make('CartPole-v0')
    net = PolicyLearner.Net()
    policy = PolicyLearner.PolicyLearner(net)
    dataHandler = DataHandler.DataHandler(policy, env)

    for i in range(1000):
        continueFlag = dataHandler.render(episodes=3)
        if not continueFlag:
            return

        reward = dataHandler.generate(episodes=2000)
        loss = dataHandler.train(batchSize=32)
        dataHandler.reset()
        print("reward:", reward, "loss:", loss.item())


if __name__ == "__main__":
    main()
