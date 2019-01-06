
import torch, sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class DataHandler:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.reset()

    def reset(self):
        self.inputs = []
        self.labels = []
        self.rewards = []

    def generate(self, episodes):
        sumReward, n = 0.0, 0
        for i in range(episodes):
            inputs, labels, rewards, totalReward = self.runEpisode()
            self.inputs.extend(inputs)
            self.labels.extend(labels)
            self.rewards.extend(rewards)
            sumReward += totalReward
            n += 1

        avgReward = sumReward / n
        return avgReward

    def train(self, batchSize):
        inputs = torch.from_numpy(np.asarray(self.inputs, dtype=np.float32))
        labels = torch.from_numpy(np.asarray(self.labels, dtype=np.int64))
        rewards = np.asarray(self.rewards, dtype=np.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        rewards = torch.from_numpy(rewards)

        dataset = TensorDataset(inputs, labels, rewards)
        dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=2)

        for i, data in enumerate(dataLoader):
            inputSamples, labelSamples, rewardSamples = data
            self.policy.learn(inputSamples, labelSamples, rewardSamples)

    def render(self, episodes=1):
        for i in range(episodes):
            self.runEpisode(doRender=True)

    def runEpisode(self, doRender=False):
        inputList, outputList = [], []
        totalReward = 0.0
        observ = self.env.reset()
        done = False
        while not done:
            if doRender:
                continueFlag = self.env.render()
                if not continueFlag:
                    sys.exit(0)

            input = np.asarray([observ], dtype=np.float32)
            output = self.policy.forward(torch.from_numpy(input))
            output = output.numpy()[0]
            observ, reward, done, info = self.env.step(output)
            inputList.extend(input)
            outputList.append(output)
            totalReward += reward

        rewardList = [totalReward] * len(outputList)
        return inputList, outputList, rewardList, totalReward
