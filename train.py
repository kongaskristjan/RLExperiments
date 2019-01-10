#!/usr/bin/python3

import Net, PolicyLearner, DataHandler, Definitions

def main():
    games = ["CartPole-v1"] * 5
    sumAvg, sumMax = 0., 0.
    for gameName in games:
        currentAvg, currentMax = testGame(gameName)
        sumAvg += currentAvg
        sumMax += currentMax

    avgReward, maxReward = sumAvg / len(games), sumMax / len(games)
    print("AvgAvg:", avgReward, " AvgMax:", maxReward)


def testGame(gameName, verbose=True):
    env = Definitions.makeSeededEnvironment(gameName)
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
        reward = dataHandler.generate(episodes=100)
        dataHandler.reset(keepSize=40000)
        dataHandler.train(batchSize=8)
        if verbose: print(gameName, "   iteration:", str(i+1) + "/" + str(nIters), "   reward:", reward,
                          "   trained on:", len(dataHandler.inputs), "    confidence multipiler:", confidenceMul)
        sumReward += reward
        maxReward = max(maxReward, reward)

    avgReward = sumReward / nIters
    if verbose:
        print("%s   Avg: %.2f   Max: %.2f" % (gameName, avgReward, maxReward))
        print()
    env.close()
    return avgReward, maxReward


if __name__ == "__main__":
    main()
