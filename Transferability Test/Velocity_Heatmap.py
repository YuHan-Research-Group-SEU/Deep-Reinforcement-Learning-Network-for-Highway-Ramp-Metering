import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

Trajectory = pd.read_csv('Trajectory_aggressive_high_volume.csv', header=None).values
Trajectory = Trajectory[(Trajectory[:, 2] > 7550) & (Trajectory[:, 2] < 13600), :]

length_interval = 25
time_interval = 5
length_num = int((13600 - 7550) / length_interval)
time_num = int(3600 / time_interval)
velocity = np.zeros([length_num, time_num])
num = np.zeros([length_num, time_num])

for i in range(Trajectory.shape[0]):
    time_idx = int(np.floor((Trajectory[i, 1] - 1) / time_interval))
    length_idx = int(np.floor((Trajectory[i, 2] - 7550) / length_interval))
    velocity[length_idx, time_idx] += Trajectory[i, 4]
    num[length_idx, time_idx] += 1

np.seterr(divide='ignore', invalid='ignore')

velocity = velocity / num
velocity[np.isnan(velocity) == 1] = 100
velocity_baseline = np.load("Velocity_baseline.npy")
rmse = 0
for i in range(242):
    for j in range(720):
        rmse += (velocity[i, j] - velocity_baseline[i, j]) ** 2
rmse = math.sqrt(rmse / (242 * 720))
print(rmse)

velocity = np.flipud(velocity)
h = sns.heatmap(velocity, cmap="rainbow", vmin=30, vmax=100, cbar=False)
cb = h.figure.colorbar(h.collections[0])

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")
plt.xticks([0, 80, 160, 240, 320, 400, 480, 560, 640, 720], [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600],
           fontproperties = 'Times New Roman', size=12)
plt.yticks([2, 42, 82, 122, 162, 202, 242], [6000, 5000, 4000, 3000, 2000, 1000, 0],
           fontproperties = 'Times New Roman', size=12)
cb.set_ticks([30, 40, 50, 60, 70, 80, 90, 100])
cb.set_ticklabels([30, 40, 50, 60, 70, 80, 90, 100], fontproperties = 'Times New Roman', size=12)
plt.xlabel("时间/(s)", fontproperties=font, size=16)
plt.ylabel("位置/(m)", fontproperties=font, size=16)
cb.set_label("速度/(km/h)", fontproperties=font, size=16)
plt.show()
