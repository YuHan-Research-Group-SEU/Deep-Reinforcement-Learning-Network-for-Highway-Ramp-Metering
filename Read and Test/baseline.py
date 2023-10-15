import gym
import traci
import numpy as np
import pandas as pd

def baseline():
    env = gym.make('RampControl-v0').unwrapped
    state = env.reset()
    ep_reward = 0
    for i in range(179):
        occupancy_judge = max(traci.inductionloop.getLastStepOccupancy('e1_5'),
                              traci.inductionloop.getLastStepOccupancy('e1_10'))
        red_time = 0
        next_state, reward, done, info = env.step(red_time)
        if occupancy_judge >= 0.12:
            ep_reward += reward
    traci.close()
    data = pd.DataFrame(env.trajectory)
    data.to_csv('Trajectory_no_control.csv', index=False, header=False)
    print(ep_reward)
    return 0

if __name__ == '__main__':
    baseline()