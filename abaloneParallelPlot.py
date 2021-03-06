import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
from math import exp

target_url = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/abalone/abalone.data")
abalone = pd.read_csv(target_url, header=None, prefix="V")
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Wt', 'Shucked Wt',
                   'Viscera Wt', 'Shell Wt', 'Rings']
summary = abalone.describe()
minRings = summary.iloc[3, [7]]  # rings是标签
maxRings = summary.iloc[7, [7]]
nrows = abalone.shape[0]
for i in range(nrows):
    dataRow = abalone.iloc[i, 1:8]
    labelColor = (abalone.iloc[i, 8] - minRings) / (maxRings - minRings)
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)
plot.xlabel("Attribute Index")
plot.ylabel("Attribute Values")
plot.show()
# 颜色从深红棕色、黄色、浅蓝色一直到深蓝色代表着环数越来越多，即年龄越来越大。
meanRings = summary.iloc[1, 7]
sdRings = summary.iloc[2, 7]
for i in range(nrows):
    dataRow = abalone.iloc[i, 1:8]
    normTarget = (abalone.iloc[i, 8] - meanRings)/sdRings
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)
plot.xlabel("Attribute Index")
plot.ylabel("Attribute Values")
plot.show()