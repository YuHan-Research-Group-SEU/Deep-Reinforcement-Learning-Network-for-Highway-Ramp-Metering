import traci
from sumolib import checkBinary
import pandas as pd
import numpy as np

redtime_set = np.load('Action Set.npy')

sumoBinary = checkBinary('sumo')
feature = []
traci.start([sumoBinary, '-c', 'simulation.sumocfg'])
occupancy1 = 0
occupancy2 = 0
ave_speed1 = 0
ave_speed2 = 0
throughout_flow = 0
input_flow = 0
for i in range(180):
    redtime = redtime_set[i]
    for j in range(20 - redtime):
        traci.simulation.step()
        time = traci.simulation.getTime()
        traci.trafficlight.setPhase('gneJ20', 0)
        occupancy1 += (traci.inductionloop.getLastStepOccupancy('e1_2') +
                  traci.inductionloop.getLastStepOccupancy('e1_3') +
                  traci.inductionloop.getLastStepOccupancy('e1_4') +
                  traci.inductionloop.getLastStepOccupancy('e1_5') +
                  traci.inductionloop.getLastStepOccupancy('e1_6')) / 5
        occupancy2 += (traci.inductionloop.getLastStepOccupancy('e1Detector_lane12_3_40') +
                  traci.inductionloop.getLastStepOccupancy('e1Detector_lane12_2_41') +
                  traci.inductionloop.getLastStepOccupancy('e1Detector_lane12_1_42') +
                  traci.inductionloop.getLastStepOccupancy('e1Detector_lane12_0_43')) / 4
        ave_speed1 += traci.edge.getLastStepMeanSpeed('lane10_2') * 3.6
        ave_speed2 += traci.edge.getLastStepMeanSpeed('lane12') * 3.6
        throughout_flow += (traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane12_3_40') +
                       traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane12_2_41') +
                       traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane12_1_42') +
                       traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane12_0_43'))
        input_flow += (traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane10_1_3_32') +
                  traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane10_1_2_33') +
                  traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane10_1_1_34') +
                  traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane10_1_0_35') +
                  traci.inductionloop.getLastStepVehicleNumber('e1_1') +
                  traci.inductionloop.getLastStepVehicleNumber('e1_0'))
        if time % 20 == 0:
            occupancy1 = occupancy1 / 20
            occupancy2 = occupancy2 / 20
            ave_speed1 = ave_speed1 / 20
            ave_speed2 = ave_speed2 / 20
            data = [time, occupancy1, occupancy2, ave_speed1, ave_speed2, throughout_flow, input_flow]
            feature.append(data)
            occupancy1 = 0
            occupancy2 = 0
            ave_speed1 = 0
            ave_speed2 = 0
            throughout_flow = 0
            input_flow = 0
    for j in range(redtime):
        traci.simulation.step()
        time = traci.simulation.getTime()
        traci.trafficlight.setPhase('gneJ20', 1)
        occupancy1 += (traci.inductionloop.getLastStepOccupancy('e1_2') +
                       traci.inductionloop.getLastStepOccupancy('e1_3') +
                       traci.inductionloop.getLastStepOccupancy('e1_4') +
                       traci.inductionloop.getLastStepOccupancy('e1_5') +
                       traci.inductionloop.getLastStepOccupancy('e1_6')) / 5
        occupancy2 += (traci.inductionloop.getLastStepOccupancy('e1Detector_lane12_3_40') +
                  traci.inductionloop.getLastStepOccupancy('e1Detector_lane12_2_41') +
                  traci.inductionloop.getLastStepOccupancy('e1Detector_lane12_1_42') +
                  traci.inductionloop.getLastStepOccupancy('e1Detector_lane12_0_43')) / 4
        ave_speed1 += traci.edge.getLastStepMeanSpeed('lane10_2') * 3.6
        ave_speed2 += traci.edge.getLastStepMeanSpeed('lane12') * 3.6
        throughout_flow += (traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane12_3_40') +
                            traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane12_2_41') +
                            traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane12_1_42') +
                            traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane12_0_43'))
        input_flow += (traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane10_1_3_32') +
                       traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane10_1_2_33') +
                       traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane10_1_1_34') +
                       traci.inductionloop.getLastStepVehicleNumber('e1Detector_lane10_1_0_35') +
                       traci.inductionloop.getLastStepVehicleNumber('e1_1') +
                       traci.inductionloop.getLastStepVehicleNumber('e1_0'))
        if time % 20 == 0:
            occupancy1 = occupancy1 / 20
            occupancy2 = occupancy2 / 20
            ave_speed1 = ave_speed1 / 20
            ave_speed2 = ave_speed2 / 20
            data = [time, occupancy1, occupancy2, ave_speed1, ave_speed2, throughout_flow, input_flow]
            feature.append(data)
            occupancy1 = 0
            occupancy2 = 0
            ave_speed1 = 0
            ave_speed2 = 0
            throughout_flow = 0
            input_flow = 0

trajectory = pd.DataFrame(feature)
trajectory.to_csv('Overall Feature_TD3.csv', header=False, index=False)
traci.close()