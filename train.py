#!/usr/bin/python3

import Net, PolicyLearner, DataHandler, Definitions

def main():
    games = ["CartPole-v1"] * 5
    sumAvg, sumMax = 0., 0.
    for envName in games:
        currentAvg, currentMax = testGame(envName)
        sumAvg += currentAvg
        sumMax += currentMax

    avgReward, maxReward = sumAvg / len(games), sumMax / len(games)
    print("AvgAvg:", avgReward, " AvgMax:", maxReward)


def testGame(envName, verbose=True):
    env = Definitions.makeSeededEnvironment(envName)
    net = Net.Net(env.observation_space, env.action_space)
    net = net.to(Definitions.device)
    policy = PolicyLearner.PolicyLearner(net)
    dataHandler = DataHandler.DataHandler(policy, env)

    sumReward = 0.
    maxReward = -1e9
    nIters = 20
    for i in range(nIters):
        dataHandler.render(episodes=3)

        confidenceMul = i
        policy.setConfidenceMul(confidenceMul)
        for j in range(10):
            reward = dataHandler.generate(episodes=10)
            dataHandler.reset(keepSize=40000)
            dataHandler.train(batchSize=8, useFraction=0.1)
        if verbose: print(envName, "   iteration:", str(i + 1) + "/" + str(nIters), "   reward:", reward,
                          "   trained on:", len(dataHandler.inputs), "    confidence multipiler:", confidenceMul)
        sumReward += reward
        maxReward = max(maxReward, reward)
        Definitions.saveModel(net, envName, i + 1, reward)

    avgReward = sumReward / nIters
    if verbose:
        print("%s   Avg: %.2f   Max: %.2f" % (envName, avgReward, maxReward))
        print()
    env.close()
    return avgReward, maxReward


if __name__ == "__main__":
    main()
