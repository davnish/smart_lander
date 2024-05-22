import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

class Q_Learning:
    def __init__(self,env,alpha,gamma,epsilon,numberEpisodes):
        '''
        env : This is the environment.
        gamma : Discount Factor.
        epsilon : Probability of choosing random action.
        alpha : learning rate the amount to be added to the new state reward.
        numberEpisodes : No of episodes to be trained.
        numberOfBins : The no of states of be divided.
        '''
        self.env=env
        self.alpha=alpha
        self.gamma=gamma 
        self.epsilon=epsilon 
        self.actionNumber=env.action_space.n # total no of action
        self.numberEpisodes=numberEpisodes

        self.sumRewardsEpisode = [] # A list of Rewards of every episodes

        self.Qmatrix = np.random.uniform(low = 0, high = 0, size=(*numberOfBins,self.actionNumber))

    
    def selectAction(self, state, index):
        
        if index<500:
            return np.random.choice(self.actionNumber)
        
        if index>7000:
            self.epsilon = 0.999*self.epsilon

        randomNumber = np.random.random()
        if randomNumber<self.epsilon:
            return np.random.choice(self.actionNumber)
        else:
            stateIndex = returnIndexState(state)
            return np.argmax(self.Qmatrix[stateIndex])
    
    def simulateEpisodes(self):
        for index in range(self.numberEpisodes):
            rewardsEpisode = []
            currState = self.env.reset()[0]
            timeStamp = 0
            terminate = False

            while not terminate and timeStamp<1000: # A timestamp restriction to all the episodes reaching more than 1000 states
                stateIndex = returnIndexState(currState)
                action = self.selectAction(currState, index)
                stateNxt, rewards, terminate, _, _ = env.step(action)
                stateNxtIndex = returnIndexState(stateNxt)
                rewardsEpisode.append(rewards)

                if not terminate:
                    error = (rewards + self.gamma*np.max(self.Qmatrix[stateNxtIndex]))-self.Qmatrix[stateIndex+(action,)]
                    self.Qmatrix[stateIndex+(action,)] += self.alpha*error
                else:
                    error = rewards - self.Qmatrix[stateIndex+(action,)]
                    self.Qmatrix[stateIndex+(action,)] += self.alpha*error
                timeStamp += 1
                currState = stateNxt

            print(f"Episode: {index}, Rewards {np.sum(rewardsEpisode)}")        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))
    
    def plotRewards(self, model_no, avg_intv):
        rwds = np.asarray(self.sumRewardsEpisode)
        score = np.sum((rwds>=200))
        hg_score = np.sum(rwds>=300)
        print(f"Ep Solved : {score}, High_Score : {hg_score}")

        solved = np.ones(numberEpisodes) * 200
        avg_rwds = np.mean(rwds.reshape(-1, avg_intv), axis = -1, keepdims=True).repeat(avg_intv, axis = 1).reshape(-1)
        rwds_df = pd.DataFrame({'Rewards': rwds, 'Average_Rewards': avg_rwds, 'Solved': solved})

        # plt.figure(figsize=(10, 10))
        plt.plot(rwds_df['Rewards'], color='blue', linewidth=2, label = 'Rewards')
        plt.plot(rwds_df['Average_Rewards'], color='orange', linestyle='dashed',linewidth=2, label = 'Avg_Rewards')
        plt.plot(rwds_df['Solved'], color='red', linestyle='dashed', linewidth=1, label = 'Solved')
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        # plt.yscale('log') 
        plt.savefig(os.path.join('results', f'Q_{model_no}'))
        plt.show()

def returnIndexState(state):

    position = state[0]
    velocity = state[1]
    angle = state[2]
    angluarVelocity = state[3]
    
    indexPosition =  np.maximum(np.digitize(position, cartPositionBin)-1, 0) # np.maximum is to hangle when np.digitize becomes 0
    indexVelocity =  np.maximum(np.digitize(velocity, cartVelocityBin)-1, 0)
    indexAngle =  np.maximum(np.digitize(angle, poleAngleBin)-1, 0)
    indexAngularVelocity =  np.maximum(np.digitize(angluarVelocity, poleAngularVelocityBin)-1, 0)

    return (indexPosition, indexVelocity, indexAngle, indexAngularVelocity)

env=gym.make('CartPole-v1')

upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin=-3
cartVelocityMax=3
poleAngleVelocityMin=-10
poleAngleVelocityMax=10
upperBounds[1]=cartVelocityMax
upperBounds[3]=poleAngleVelocityMax
lowerBounds[1]=cartVelocityMin
lowerBounds[3]=poleAngleVelocityMin

# Defining the parameters for state discretization
numberOfBinsPosition=30
numberOfBinsVelocity=30
numberOfBinsAngle=30
numberOfBinsAngleVelocity=30
numberOfBins=[numberOfBinsPosition,numberOfBinsVelocity,numberOfBinsAngle,numberOfBinsAngleVelocity]

cartPositionBin = np.linspace(lowerBounds[0], upperBounds[0], numberOfBins[0])
cartVelocityBin = np.linspace(lowerBounds[1], upperBounds[1], numberOfBins[1])
poleAngleBin = np.linspace(lowerBounds[2], upperBounds[2], numberOfBins[2])
poleAngularVelocityBin = np.linspace(lowerBounds[3], upperBounds[3], numberOfBins[3])


if __name__ == '__main__':

    
    #Hyperparameters##
    model_no = 4
    numberEpisodes=20000
    alpha=0.1
    gamma=1
    epsilon=0.2
    ##################
    
    Q1=Q_Learning(env,alpha,gamma,epsilon,numberEpisodes)
    start = time.time()
    Q1.simulateEpisodes()
    end = time.time()
    print(f'Time: {end-start}')
    Q1.plotRewards(model_no, avg_intv = 800)
    np.save(os.path.join('models', f'Qmatrix_{model_no}'), Q1.Qmatrix)

    