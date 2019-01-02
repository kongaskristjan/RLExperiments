#!/usr/bin/python3

import Net, PolicyLearner, DataHandler, Definitions

def main():
    games = ["CartPole-v1"]
    sumReward = 0.
    for gameName in games:
        sumReward += testGame(gameName)
    avgReward = sumReward / len(games)
    print("All games average reward:", avgReward)


def testGame(gameName, verbose=True):
    env = Definitions.makeSeededEnvironment(gameName)
    net = Net.Net()
    net = net.to(Definitions.device)
    policy = PolicyLearner.PolicyLearner(net)
    dataHandler = DataHandler.DataHandler(policy, env)

    sumReward = 0.
    nIters = 10
    for i in range(nIters):
        dataHandler.render(episodes=3)

        reward = dataHandler.generate(episodes=2000)
        dataHandler.train(batchSize=32)
        dataHandler.reset()
        if verbose: print(gameName, "   iteration:", str(i+1) + "/" + str(nIters), "   reward:", reward)
        sumReward += reward

    avgReward = sumReward / nIters
    if verbose: print(gameName, "   average reward:", avgReward)
    return avgReward


if __name__ == "__main__":
    main()
