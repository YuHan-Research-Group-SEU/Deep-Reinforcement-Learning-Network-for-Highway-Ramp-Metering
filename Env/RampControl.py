import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
import sys
import os
import traci
from sumolib import checkBinary
import time

class RampControlEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(np.array([0]), np.array([20]), dtype=np.int)
        self.observation_space = spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(7,1))
        self.seed()
        self.state = None
        self.trajectory = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def save_trajectory(self):
        time = traci.simulation.getTime()
        veh_id = traci.vehicle.getIDList()
        ramp_edge = ['lane13', 'lane14', 'lane15_1', 'lane16', ':gneJ16_0', ':genJ18_0', ':gneJ20_0', 'lane15_2']
        if len(veh_id) == 0:
            pass
        else:
            for i in veh_id:
                edge = traci.vehicle.getRoadID(i)
                if edge in ramp_edge:
                    continue
                position = traci.vehicle.getPosition(i)
                speed = traci.vehicle.getSpeed(i) * 3.6
                trj = [i, time, position[0], position[1], speed]
                self.trajectory.append(trj)

    def reset(self):
        sumoBinary = checkBinary('sumo')
        traci.start([sumoBinary, '-c', 'simulation.sumocfg', '--statistic-output', 'output.xml', '--duration-log.statistics'])
        inflow_mainline = 0
        inflow_ramp = 0
        mainline_id = []
        ramp_id = []
        halting_total = []
        for t in range(20):
            traci.trafficlight.setPhase('gneJ20', 0)
            traci.simulationStep()
            inflow_mainline, mainline_id = self.get_mainline_inflow(inflow_mainline, mainline_id)
            inflow_ramp, ramp_id = self.get_ramp_inflow(inflow_ramp, ramp_id)
            halting_total.append(self.get_queue_num())
            #self.save_trajectory()
        queue_length = halting_total[-1]
        self.state = self.get_observation(inflow_mainline, inflow_ramp, queue_length)

        return self.state

    def step(self, redtime):
        reward = 0
        inflow_mainline = 0
        inflow_ramp = 0
        mainline_id = ()
        ramp_id = ()
        halting_total = []
        done = False
        for t in range(20 - redtime):
            traci.trafficlight.setPhase('gneJ20', 0)
            traci.simulationStep()
            inflow_mainline, mainline_id = self.get_mainline_inflow(inflow_mainline, mainline_id)
            inflow_ramp, ramp_id = self.get_ramp_inflow(inflow_ramp, ramp_id)
            halting_total.append(self.get_queue_num())
            reward = self.get_reward_traveltime(reward)
            #self.save_trajectory()
        for t in range(redtime):
            traci.trafficlight.setPhase('gneJ20', 1)
            traci.simulationStep()
            inflow_mainline, mainline_id = self.get_mainline_inflow(inflow_mainline, mainline_id)
            inflow_ramp, ramp_id = self.get_ramp_inflow(inflow_ramp, ramp_id)
            halting_total.append(self.get_queue_num())
            reward = self.get_reward_traveltime(reward)
            #self.save_trajectory()
        queue_length = halting_total[-1]
        obs = self.get_observation(inflow_mainline, inflow_ramp, queue_length)
        info = {}

        return obs, reward, done, info

    def get_observation(self, inflow1, inflow2, queue_length):
        density1 = traci.edge.getLastStepVehicleNumber('lane10_1') / 181 * 1000 / 4
        density2 = traci.edge.getLastStepVehicleNumber('lane10_2') / 160 * 1000 / 5
        speed_1 = traci.edge.getLastStepMeanSpeed('lane10_1')
        speed_2 = traci.edge.getLastStepMeanSpeed('lane10_2')

        self.state = np.array([density1, density2, inflow1, inflow2, speed_1, speed_2, queue_length], dtype=np.float32)

        return self.state

    def get_reward_traveltime(self, reward):
        func_value = -(traci.edge.getLastStepVehicleNumber('lane10_2') - 27) ** 2
        reward += func_value

        return reward

    def get_mainline_inflow(self, throughout=0, veh_id=()):
        id_in_detector = traci.inductionloop.getLastStepVehicleIDs('e1Detector_lane10_1_3_32') + \
                         traci.inductionloop.getLastStepVehicleIDs('e1Detector_lane10_1_2_33') + \
                         traci.inductionloop.getLastStepVehicleIDs('e1Detector_lane10_1_1_34') + \
                         traci.inductionloop.getLastStepVehicleIDs('e1Detector_lane10_1_0_35')
        new_n_detector = list(set(id_in_detector) - set(veh_id))
        throughout = throughout + len(new_n_detector)

        return throughout, id_in_detector

    def get_ramp_inflow(selfself, throughout=0, veh_id=()):
        id_in_detector = traci.inductionloop.getLastStepVehicleIDs('e1_0') + \
                         traci.inductionloop.getLastStepVehicleIDs('e1_1')
        new_n_detector = list(set(id_in_detector) - set(veh_id))
        throughout = throughout + len(new_n_detector)

        return throughout, id_in_detector


    def get_queue_num(self):
        queue_num = 0
        vehicles_on_ramp = traci.edge.getLastStepVehicleIDs('lane15_1') +\
                           traci.edge.getLastStepVehicleIDs('lane16') +\
                           traci.edge.getLastStepVehicleIDs(':genJ18_0')
        for id in vehicles_on_ramp:
            if traci.vehicle.getSpeed(id) < 3:
                queue_num += 1
        return queue_num