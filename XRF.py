import pandas as pd
import matplotlib.pyplot as plt
import re, glob

def read_add_plot(file,ax,label):
    data = pd.read_csv(file,header=20)
    ax.scatter(data['Channel#'],data['Intensity'],marker='.',label=label)

# file = r'Data\PS1A_120s_40kV_20pA_nofill_air.csv'
fig,(ax1,ax2) = plt.subplots(1,2)
files = glob.glob(r'Data\PS1*40kV*.csv')
for file in files:
    label=re.search('PS1[A-Z]*',file).group(0)
    read_add_plot(file,ax1,label)
ax1.set_yscale('log')
ax1.legend()
plt.show()