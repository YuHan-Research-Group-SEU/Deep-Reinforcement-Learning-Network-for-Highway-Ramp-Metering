import TD3
import sys
import os
import gym
import traci
import numpy as np
import pickle
import copy
from xml.dom.minidom import parse

#调用TraCI的语句
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

#Hyper parameters
LR_ACTOR = 0.00001
LR_CRITIC = 0.00002
GAMMA = 0.99
MEMORY_SIZE = 2048
BATCH_SIZE = 256
MAX_EPISODE = 400
TAU = 0.02
VAR = 0.4
POLICY_NOISE = 0.2
NOISE_CLIP = 0.3
POLICY_FREQ = 150
UPDATE_TIME = 2
env = gym.make('RampControl-v0').unwrapped
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.shape[0]
RED_TIME_BOUND = env.action_space.high[0]
ACTOR_FILEPATH = 'Actor.data'
CRITIC_FILEPATH = 'Critic.data'

#Main function
if __name__ == "__main__":
    agent = TD3.TD3Agent(MEMORY_SIZE, N_STATES, LR_ACTOR, LR_CRITIC, BATCH_SIZE, GAMMA, TAU, VAR, POLICY_NOISE, NOISE_CLIP, POLICY_FREQ)
    Reward = np.zeros((MAX_EPISODE, 1))
    ATT = np.zeros((MAX_EPISODE, 1))
    actor_param = {}
    critic_param = {}

    state = env.reset()
    for i in range(179):
        occupancy_judge = max(traci.edge.getLastStepOccupancy('lane10_1'),
                              traci.edge.getLastStepOccupancy('lane10_2'))
        red_time = 0
        action = np.array([0])
        next_state, reward, done, info = env.step(red_time)
        if occupancy_judge >= 0.12:
            agent.memory.push(state, action, reward, next_state)
        state = next_state
    traci.close()

#Train-process
    for episode in range(MAX_EPISODE):
        state = env.reset()
        for _ in range(179):
            occupancy_judge = max(traci.edge.getLastStepOccupancy('lane10_1'),
                                  traci.edge.getLastStepOccupancy('lane10_2'))
            if occupancy_judge < 0.12:
                red_time = 0
                next_state, reward, done, info = env.step(red_time)
            else:
                action = agent.choose_action(state)
                action = np.clip(np.random.normal(action, VAR), -1, 1)
                red_time = abs(int(RED_TIME_BOUND * action[0]))
                next_state, reward, done, info = env.step(red_time)
                agent.memory.push(state, action, reward, next_state)
            if agent.memory.memory_counter >= MEMORY_SIZE:
                VAR = max(VAR - 0.005, 0.1)
                for i in range(UPDATE_TIME):
                    agent.learn()
            state = next_state
        traci.close()

#Test-Process
        state_test = env.reset()
        ep_reward = 0
        for i in range(179):
            occupancy_judge = max(traci.edge.getLastStepOccupancy('lane10_1'),
                                  traci.edge.getLastStepOccupancy('lane10_2'))
            if occupancy_judge < 0.12:
                red_time = 0
                next_state_test, reward, done, info = env.step(red_time)
            else:
                action_test = agent.choose_action(state_test)
                red_time = abs(int(RED_TIME_BOUND * action_test[0]))
                next_state_test, reward, done, info = env.step(red_time)
                ep_reward += reward
            state_test = next_state_test
        traci.close()

        Reward[episode, 0] = ep_reward
        domTree = parse('output.xml')
        rootNode = domTree.documentElement
        list = rootNode.getElementsByTagName('vehicleTripStatistics')[0]
        duration = float(list.getAttribute('duration'))
        ATT[episode, 0] = duration
        print("Ep:", episode, "|Reward:", ep_reward, "|ATT:", duration)

        actor_param[episode]= copy.deepcopy(agent.actor_eval.state_dict())
        critic_param[episode] = copy.deepcopy(agent.critic_eval.state_dict())

#Save Data
    f = open(ACTOR_FILEPATH, 'wb')
    pickle.dump(actor_param, f)
    f.close()

    f = open(CRITIC_FILEPATH, 'wb')
    pickle.dump(critic_param, f)
    f.close()

    np.save('Ep_Reward.npy', Reward)
    np.save('Average Travel Time.npy', ATT)