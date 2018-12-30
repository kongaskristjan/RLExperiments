#!/usr/bin/python3

import numpy as np
import gym, torch
import PolicyLearner

def main():
    env = gym.make('CartPole-v0')
    policy = PolicyLearner.PolicyLearner(PolicyLearner.Net())

    for i in range(10):
        observ = env.reset()
        done = False
        while not done:
            if not env.render():
                return

            input = np.asarray([observ], dtype=np.float32)
            input = torch.from_numpy(input)
            output = policy.forward(input)
            action = output.numpy()[0]
            observ, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()
