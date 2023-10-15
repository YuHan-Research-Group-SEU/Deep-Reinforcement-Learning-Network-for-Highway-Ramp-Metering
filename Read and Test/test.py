import gym
import traci
import torch
import numpy as np
import pickle
import TD3
import pandas as pd

env = gym.make('RampControl-v0').unwrapped
N_STATES = env.observation_space.shape[0]
RED_TIME_BOUND = env.action_space.high[0]

f = open('Actor.net', 'rb')
param = pickle.load(f)
f.close()
actor = TD3.Actor(N_STATES)
actor.load_state_dict(param)

state = env.reset()
ep_reward = 0
action_set = [0]
for i in range(269):
    state = torch.tensor(state, dtype=torch.float32)
    occupancy_judge = max(traci.edge.getLastStepOccupancy('lane10_1'),
                          traci.edge.getLastStepOccupancy('lane10_2'))
    if occupancy_judge >= 0.12:
        action = actor(state).detach().numpy()
        red_time = abs(int(RED_TIME_BOUND * action[0]))
    else:
        red_time = 0
    action_set.append(red_time)
    next_state, reward, done, info = env.step(red_time)
    state = next_state
traci.close()
trajectory = pd.DataFrame(env.trajectory)
trajectory.to_csv('Trajectory_TD3.csv', header=False, index=False)
action_set = np.array(action_set)
np.save('Action Set.npy', action_set)