import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import gym

env = gym.make('RampControl-v0').unwrapped
N_STATES = env.observation_space.shape[0]

reward = np.load('Ep_Reward.npy').reshape([400])
ATT = np.load('Average Travel Time.npy').reshape([400])
reward_idx = np.argmax(reward)
ATT_idx = np.argmin(ATT)
f = open('Actor.data', 'rb')
param = pickle.load(f)
f.close()
actor = param[399]
f = open('Actor.net', 'wb')
pickle.dump(actor, f)
f.close()

episode = np.arange(400).reshape([400])
# plt.plot(episode, reward, color='#5F97D2')
plt.plot(episode, ATT, color='#5F97D2')

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")
plt.xlabel('仿真运行轮次', fontproperties=font, size=14)
# plt.ylabel('单轮次累计奖励值', fontproperties=font, size=14)
plt.ylabel('车辆平均行驶时间/(s)', fontproperties=font, size=14)
plt.grid(color='#A9B8C6', linestyle=':')

plt.xlim([0, 400])
# plt.ylim([-190000, -20000])
plt.ylim([315, 370])
plt.xticks([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400], fontproperties = 'Times New Roman', size=12)
# plt.yticks([-180000, -160000, -140000, -120000, -100000, -80000, -60000, -40000, -20000], fontproperties = 'Times New Roman', size=12)
plt.yticks([315, 325, 335, 345, 355, 365], fontproperties = 'Times New Roman', size=12)
plt.show()