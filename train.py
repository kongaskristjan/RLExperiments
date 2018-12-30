#!/usr/bin/python3

import gym

def main():
    env = gym.make('CartPole-v0')
    env.reset()
    print(env.action_space)
    print(env.observation_space)
    while env.render():
        env.step(env.action_space.sample())  # take a random action


if __name__ == "__main__":
    main()
