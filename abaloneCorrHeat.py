import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
import seaborn as sns
target_url = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/abalone/abalone.data")
abalone = pd.read_csv(target_url, header=None, prefix="V")
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                   'Viscera weight', 'Shell weight', 'Rings']
corMat = DataFrame(abalone.iloc[:, 1:9].corr())
print(corMat)
# plot.pcolor(corMat)
# plot.show()
colormap = plot.cm.viridis
plot.figure(figsize=(12, 12))
sns.heatmap(corMat.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plot.show()