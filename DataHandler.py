
import torch, sys, random
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class DataHandler:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.reset()

    def reset(self, keepSize = 0):
        if 'inputs' not in dir(self):
            self.inputs = []
            self.labels = []
            self.rewards = []

        # Shuffle all lists with same order
        if len(self.inputs) > 0:
            sampleEfficiency = np.abs((self.rewards - np.mean(self.rewards)) / np.std(self.rewards))
            sampleEfficiency += np.random.normal(scale=0.12, size=sampleEfficiency.shape[0])
            indices = list(np.argsort(sampleEfficiency))
            np.random.shuffle(indices)
            self.inputs = [ self.inputs[i] for i in indices ]
            self.labels = [ self.labels[i] for i in indices ]
            self.rewards = [ self.rewards[i] for i in indices ]


        throwAway = max(0,len(self.inputs)-keepSize)
        self.inputs = self.inputs[throwAway:]
        self.labels = self.labels[throwAway:] 
        self.rewards = self.rewards[throwAway:] 

    def generate(self, episodes):
        sumReward = 0.0
        for i in range(episodes):
            inputs, labels, rewards, totalReward = self.runEpisode()
            self.inputs.extend(inputs)
            self.labels.extend(labels)
            self.rewards.extend(rewards)
            sumReward += totalReward

        avgReward = sumReward / episodes
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

    def runEpisode(self, doRender=False, discountFactor=0.97):
        inputList, outputList = [], []
        totalReward = 0.0
        observ = self.env.reset()
        done = False
        rewardList = []
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
            rewardList.append(reward)
            totalReward += reward

        for i in range(len(outputList)-2,-1,-1):
            rewardList[i] += rewardList[i+1]*discountFactor
        return inputList, outputList, rewardList, totalReward
