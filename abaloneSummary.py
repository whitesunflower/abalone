import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot

target_url = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/abalone/abalone.data")
abalone = pd.read_csv(target_url, header=None, prefix="V")
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
                   'Shell weight', 'Rings']
print(abalone.head())  # 打印头5行
print(abalone.tail())  # 打印后5行
# print(abalone.describe())  # 打印基本统计量
summary = abalone.describe()
array = abalone.iloc[:, 1:9].values
boxplot(array)
plot.xlabel("Attribute Index")
plot.ylabel("Quartile Ranges")
show()

# 除去最后一列数据再绘制箱线图
array2 = abalone.iloc[:, 1:8].values
boxplot(array2)
plot.xlabel("Attribute Index")
plot.ylabel("Quartile Ranges")
show()

# 画图前先将属性值都归一化
abaloneNormalized = abalone.iloc[:, 1:9]
for i in range(8):
    mean = summary.iloc[1, i]   # summary是数据集基本统计量，读到每一列数据的平均值
    sd = summary.iloc[2, i]     # 读到每一列数据的标准差
    abaloneNormalized.iloc[:, [i]] = (abaloneNormalized.iloc[:, [i]] - mean) / sd

array3 = abaloneNormalized.values
boxplot(array3)
plot.xlabel("Attribute Index")
plot.ylabel("Quartile Ranges - Normalized ")
show()