import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def fund(x, a, b):
    return a * np.exp(x) + b

x2 = np.arange(0, 25, 0.1)
Performance = pd.read_excel('Performance.xlsx')
mismatch = Performance['差异度'].values
delay_relative_decrease = Performance['优化幅度'].values * 100
popt, pcov = curve_fit(fund, mismatch, delay_relative_decrease)
y2 = [fund(xx,popt[0],popt[1]) for xx in x2]
l1, = plt.plot(mismatch, delay_relative_decrease, marker='*', color='#D76364', linewidth=3)
l2, = plt.plot(x2, y2, color='#5F97D2', linewidth=3)
plt.legend(handles=[l1, l2], labels=['实际曲线', '趋势线'], prop={'family': 'Simsun', 'size':13})
plt.xlabel('差异程度/（%）', fontproperties='simsun', size=16)
plt.ylabel('车均延误优化率/(%)', fontproperties='simsun', size=16)
plt.xlim([0, 25])
plt.ylim([0, 40])
plt.xticks([0, 5, 10, 15, 20, 25], fontproperties = 'Times New Roman', size=12)
plt.yticks([0, 10, 20, 30, 40], fontproperties = 'Times New Roman', size=12)
plt.grid(color='#A9B8C6', linestyle=':')
plt.show()