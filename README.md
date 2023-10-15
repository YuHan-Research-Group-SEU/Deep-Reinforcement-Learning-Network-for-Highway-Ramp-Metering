# Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering
This repository is aimed to provide a procedure for training an agent for highway ramp metering and testing its transferability depending on the environment built by OpenAI Gym and Eclipse SUMO.

## Prepare the simulation environment
The environment is built based on the open micro traffic simulation software Eclipse SUMO (Simulation of Urban Mobility). To run the simulation in coding scripts, ensure the following denpendencies have been installed on your system:
* [Python](https://www.python.org/) 3.10.8
* [SUMO](https://eclipse.dev/sumo/) 1.14.1
* [Pytorch](https://pytorch.org/) 1.13.1 + cu116

In this work, 25 simulation scenarios with different micro driving characteristics and traffic demands were established for a certain highway network. Among all these scenarios, 
[rongwu_route_II_III.rou.xml](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Simulation/rongwu_route_II_III.rou.xml) is the baseline environment.  

The first Roman digit in the rou file name represents its car-following and lane-changing model (the corresponding relationship is shown in the table below), and the second Roman digit represent the traffic demand in this environment (the larger the value, the greater the load).
| First Roman Digit      | Car-following Model    | Lane-changing Model    |
| :--------------------: | :--------------------: | :--------------------: |
| I                      | Krauss (aggressive)    | LC2013 (aggressive)    |
| II                     | Krauss (standard)      | LC2013 (standard)      |
| III                    | Krauss (conservative)  | LC2013 (conservative)  |
| IV                     | IDM (standard)         | LC2013 (standard)      |
| V                      | EIDM (standard)        | LC2013 (standard)      |

Once you have selected a certain rou file, it will work together with [net](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Simulation/rongwu.net.xml)
and [sumocfg](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Simulation/simulation.sumocfg) to form a storage environment for agent training and testing. 

## How to train a new agent
Besides the pre-trained agent in our work, the environment and training frameworks are also provided for training a new agent for ramp metering:  
* [RampControl.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Env/RampControl.py) is the interactive enviorment for agent which should be added to gym library.
* [TD3.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Model%20and%20Training/TD3.py) is the deep reinforcement learning algorithm in this work.
* [run.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Model%20and%20Training/run.py) is the executing scripts for agent training.

After preparing a specific simulation environment and configuring [RampControl.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Env/RampControl.py) in gym library, you just 
need to set some hyper-parameters and execute [run.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Model%20and%20Training/run.py), the network, total reward and average travel time of each episode will be saved 
in [Actor.data](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Results/Actor.data), [Critic.data](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Results/Critic.data), 
[Ep_reward.npy](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Results/Ep_Reward.npy) and [Average Travel Time.npy](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Results/Average%20Travel%20Time.npy).

Once you have a series of network in training process, [read.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Read%20and%20Test/read.py) will plot the reward in training process and select the final network 
[Actor.net](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Results/Actor.net) as the result of agent.

## How to evaluate the perfomence pre-trained agent
You can get some indicies of controling effects of the pre-trained agent by executing [test.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Read%20and%20Test/test.py). Record the results of Statistics (Avg) shown in debug console (especially ‘Speed’, ‘Duration’ and ‘Timeloss’) 
and you will get the performace of the agent in original environment. Note that you can modify the annotations in [RampControl.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Env/RampControl.py) to decide whether to save the trajectory data of all vehicles in this test.  

In addition to the performance in original environment, we also provide the framework the evaluate the transferability of pre-trained agent. Change the ‘route-files value’ in ‘simulation.sumocfg’  into a different rou file, and a different simulation environment has been constructed. Then 
execute [test.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Read%20and%20Test/test.py) and you will get the performace of the agent in a different environment. 

Repeat the following steps for each rou file and save all their trajectory files (.csv), execute [Velocity_Heatmap.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Transferability%20Test/Velocity_Heatmap.py) to draw the velocity heatmap in each test and calculate RMSE between this map and 
that in original environment. Recoding the performance and RMSE in [performance.xlsx](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Transferability%20Test/Performance.xlsx) and run [Transfer_Figure.py](https://github.com/YuHan-Research-Group-SEU/Deep-Reinforcement-Learning-Network-for-Highway-Ramp-Metering/blob/main/Transferability%20Test/Transfer_Figure.py) 
result of the transferability of pre-trained agent will be ploted in a figure (x-axis is the difference, y-axis is the performence).
