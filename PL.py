import numpy as np 
import matplotlib.pyplot as plt
import glob,re
import pandas as pd
# from functions import singlePeakFit
from lmfit.models import GaussianModel

########
from cycler import cycler
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
# import matplotlib.parkpalettes as pp
# # mpl.rcParams['axes.prop_cycle'] = cycler(color=pp.park_palettes_hex["Denali"])

# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 14}
# mpl.rc('font', **font)
#################

hc = 1.2E3
def gauss_function(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gaussian(xdata,ydata):
    model = GaussianModel()
    params = model.guess(ydata, x=xdata)
    # params = [0.9,100,2]
    out = model.fit(ydata, params=params, x=xdata)
    c,a = out.params['center'].value, out.params['amplitude'].value

    yfit = out.best_fit
    return yfit, c, a


# # files = glob.glob(r'C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.05 PL\Sn2S*.txt')
# files_3 = [r'C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.05 PL\PS1AS2.txt',r'C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.05 PL\Sn2S_2.txt']
# files_5 = [r'C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.05 PL\Sn2S_3.txt',r'C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.05 PL\Sn2S_4.txt']
fig,ax = plt.subplots(1,1)

# wavelength=[]
# counts=[]
# cps=[]
# for file in files_3:
#     data = np.genfromtxt(file, delimiter=',')
#     wavelength.extend(data[:,0].tolist())
#     counts.extend(data[:,1].tolist())

# cps = [x//3 for x in counts]

# counts = []
# for file in files_5:
#     data = np.genfromtxt(file, delimiter=',')
#     wavelength.extend(data[:,0].tolist())
#     counts.extend(data[:,1].tolist())

# cps.extend(x//5 for x in counts)
# ev = [hc/w for w in wavelength]
# ax.scatter(ev,cps,marker=".",label='Sn2S')

# files_3 = [r'C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.05 PL\Pb1_Sn1_S_1.txt',r'C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.05 PL\Pb1_Sn1_S_2.txt',r'C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.05 PL\Pb1_Sn1_S_3.txt']
# files_5 = [r'C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.05 PL\Pb1_Sn1_S_5.txt']
# wavelength=[]
# counts=[]
# cps = []
# for file in files_3:
#     data = np.genfromtxt(file, delimiter=',')
#     wavelength.extend(data[:,0].tolist())
#     counts.extend(data[:,1].tolist())

# cps = [x//5 for x in counts]

# counts = []
# for file in files_5:
#     data = np.genfromtxt(file, delimiter=',')
#     wavelength.extend(data[:,0].tolist())
#     counts.extend(data[:,1].tolist())

# cps.extend(x//5 for x in counts)
# ev = [hc/w for w in wavelength]
# ax.scatter(ev,cps,marker=".",label='Pb1Sn3S')


files = glob.glob(r'C:\Users\jarab\DataProcessing\Data Olivia\Sn1S1n*.txt')
wavelength=[]
counts=[]
cps = []
for file in files:
    print(file)
    data = np.genfromtxt(file, delimiter=',')
    wavelength.extend(data[:,0].tolist())
    counts.extend(data[:,1].tolist())

cps = [x//10 for x in counts]
ev = [hc/w for w in wavelength]
ax.scatter(ev,cps,marker=".",label='Sn1S')
yfit,c,a = fit_gaussian(np.array(ev),np.array(cps))
ax.plot(ev,yfit,label=f'Sn1S1 Gaussian fit with center {c} and amplitude {a}')

# files = glob.glob(r'C:\Users\jarab\DataProcessing\Data Olivia\Sn2S*.txt')
# wavelength=[]
# counts=[]
# cps = []
# for file in files:
#     print(file)
#     data = np.genfromtxt(file, delimiter=',')
#     wavelength.extend(data[:,0].tolist())
#     counts.extend(data[:,1].tolist())
# cps = [x//10 for x in counts]
# ev = [hc/w for w in wavelength]
# ax.scatter(ev,cps,marker='.',label='Sn2S2')
# yfit,c,a = fit_gaussian(np.array(ev),np.array(cps))
# ax.plot(ev[1:],yfit[1:],label=f'Sn2S2 Gaussian fit with center {c} and amplitude {a}')

# files = glob.glob(r'C:\Users\jarab\DataProcessing\Data Olivia\Sn3S*.txt')
# wavelength=[]
# counts=[]
# cps = []
# for file in files:
#     print(file)
#     data = np.genfromtxt(file, delimiter=',')
#     wavelength.extend(data[:,0].tolist())
#     counts.extend(data[:,1].tolist())
# cps = [x//10 for x in counts]
# ev = [hc/w for w in wavelength]
# ax.scatter(ev,cps,marker='.',label='Sn3S3')
# yfit,c,a = fit_gaussian(np.array(ev),np.array(cps))
# ax.plot(ev[1:],yfit[1:],label=f'Sn3S3 Gaussian fit with center {c} and amplitude {a}')

# files = glob.glob(r'C:\Users\jarab\DataProcessing\Data Olivia\background*.txt')
# wavelength=[]
# counts=[]
# cps = []
# for file in files:
#     print(file)
#     data = np.genfromtxt(file, delimiter=',')
#     wavelength.extend(data[:,0].tolist())
#     counts.extend(data[:,1].tolist())

# cps = [x//10 for x in counts]
# ev = [hc/w for w in wavelength]
# ax.scatter(ev,cps,marker=".",label='background')

############ Fit with a Gaussian ############
# x=np.array(wavelength)
# y=np.array(cps)

# model = GaussianModel()
# params = model.guess(y, x=x)
# out = model.fit(y, params=params, x=x)
# c,a = out.params['center'].value, out.params['amplitude'].value

# yfit = out.best_fit
# ax.scatter(x,yfit,marker='.',label=f'Fit Center = {c}; Fit Ampl. = {a}')
#######################################################
plt.legend()
plt.show()

