import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from data_plotters /import plotSLACxrd, plotXRD
import glob,re
from fileIO import getPanalyticalXRD, slacXRDraw, rigakuXRD
# from cycler import cycler
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

# import matplotlib.parkpalettes as pp

# plt.style.use('publication')
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 24}

# mpl.rcParams['axes.prop_cycle'] = cycler(color=pp.park_palettes_hex["GreatBasin"])

# mpl.rc('font', **font)
fig,(ax0,ax1,ax2,ax3) = plt.subplots(nrows=4, sharex=True,gridspec_kw={'height_ratios': [3,1,1,1]})
plt.subplots_adjust(hspace=0)

##### Reference Patterns###############
# df = pd.read_csv(r"C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.02 XRD\SnS_reflections.csv",sep=',')
# df1 = df[df['ID(λ)']==1]
# twotheta = df1['2θ']
# intensity = df1['I']
# ax1.stem(twotheta,intensity,markerfmt='',label='SnS Reference Kα1')
# df = pd.read_csv(r"C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.02 XRD\PbS_reflections.csv",sep=',')
# df1 = df[df['ID(λ)']==1]
# twotheta = df1['2θ']
# intensity = df1['I']
# ax2.stem(twotheta,intensity,markerfmt='',label='PbS Reference Kα1')
# df = pd.read_csv(r"C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.02 XRD\SnS2_reflections.csv",sep=',')
# df1 = df[df['ID(λ)']==1]
# twotheta = df1['2θ']
# intensity = df1['I']
# ax3.stem(twotheta,intensity,markerfmt='',label='SnS2 Reference Kα1')


filelist = [ 
           r"C:\Users\JaramilloGroup\DataProcessing\Data\PS1AS7_2tw_20_80_12min_1.xrdml",
           r"C:\Users\JaramilloGroup\DataProcessing\Data\PS1BS7_2tw_20_80_12min_1.xrdml",
           r"C:\Users\JaramilloGroup\DataProcessing\Data\PS1CS7_2tw_20_80_12min_1.xrdml"
   
        #    r"C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.02 XRD\Sn1S_wSiO2_2tw_survey.ras"
            ]
# filelist = glob.glob(r"C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.02 XRD\Sn2S*.ras")
# # # #  files = glob.glob(r"\Users\oschn\Dropbox\RNO Data\XRD\ppms\*")
# plotz = plotXRD(filelist,  offset=1, together=True, save=False,legend=False)
# plt.show()
# # # # plotXRD(filelist,together=False)
# for file in glob.glob(r"C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.02 XRD\Sn*_12min*"):
m=1
for file in filelist:
    # data = rigakuXRD(file)
    # sample=re.search(r"XRD\\(Pb1)?Sn3S2?_2tw",file).group(0)
    if file[-1]=='l':
        tt,cps = getPanalyticalXRD(file)
    elif file[-1]=='s':
        data = rigakuXRD(file)
        tt = data[:,0]
        cps = data[:,1]
    name = re.search(r'PS1[A-Z]*S7',file).group(0)
    # name=re.search("(Pb|Sn)\d(a|b|c)*S*\d*",file).group(0)
    # x = np.array(data[0])
    # # y = np.array(data[:,1])
    # z = np.array(data[:,1])
#     sub_int = y[np.argwhere(x>47.5)]
#     sub_2t = x[np.argwhere(x>47.5)]
#     epi_2t = sub_2t[np.argmax(sub_int)]
#     epi_int = sub_int[np.argmax(sub_int)]
#     print(epi_2t)
#     if sample == "C1-02654":
#         label=" 10 nm"
#     elif sample == "C1-02653":
#         label=" 5 nm"
#     elif sample == "C1-02662":
#         label=" 25 nm"
#     if name:
#         label = name.group(0)[0]=="P" 
#         plt.plot(x,y,label=name)
#     else:
    ax0.plot(tt,cps*m,label=' ',marker="")
    m=m*1
   
    
    # ax0.set_yscale('log')
    
plt.xlim((20,80))


# df = pd.read_csv(r"C:\Users\oschn\Dropbox (MIT)\20-29 TLP PbSnSSe\22 Data\22.02 XRD\SnS2_reflections.csv",sep=',')
# df2 = df[df['ID(λ)']==1]
# twotheta = df2['2θ']
# intensity = df2['I']
# ax3.stem(twotheta,intensity,markerfmt='',label='SnS2 Reference Kα1')
# ax3.set_xlim([20,80])
# Set tick positions and labels for x-axis
# ax.set_xticks(range(24,90, 2))  # Ticks every 5 units
# Set tick labels for x-axis
# ax.set_xticklabels(range(24,90, 2))
for ax in (ax1,ax0,ax2,ax3):
    ax.legend()
    # ax.grid(True)
    ax.set_yscale('log')

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(20, 81, 5)
    minor_ticks = np.arange(20, 81, 1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(True,which='both',axis='x')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

plt.xlabel('2θ')
ax0.set_ylabel('Intensity')

plt.show()
# # vfiles = glob.glob(r'/Users/oschnebl/Dropbox/RNO Data/XRD/targets/N*.txt')
# colors = ["g","midnightblue","darkorange","firebrick"]
# i=0
# for vfile in vfiles:
#     print(vfile)
#     df = pd.read_csv(vfile,sep='\s+')
#     phase = re.search('N\w*_vesta',vfile).group(0)[0:-6]
#     print(phase,df.head())
#     xv = df['2θ']
#     yv = df['I']*3.775
#     plt.scatter(xv,yv,marker="+",c=colors[i],label=phase)
#     i+=1
# plt.legend()
# plt.show()


# from brooks_master.RSM import RSM

# ##Data

# rsm1 = {   'RSM_file':'/Users/olivia/Dropbox/RNO Data/C1_2590_2598/RSM/C1-2590_NGO_0b26_RSM_1D_3h_zoomed/C1-2590_b206_RSM_1D_3hr_zoomed.ras',
#             'RSM':RSM('NGO', 'LNO', (0,2,6)),
#             'label':'C3-02286'}

# # rsm2 = {   'RSM_file':'/Users/olivia/Dropbox/RNO Data/ENO from Data/C1-02662_NGO_0b26_RSM_1D_6h.ras',
# #             'RSM':RSM('NGO', 'LNO', (0,2,6)),
# #             'label':'C3-02286'}

# # rsm1 = {   'RSM_file':'/Users/oschnebl/Library/CloudStorage/Box-Box/RNO Neuromorphic LDRD/Data/C1/02598/C1-2598_b206_RSM_1D_3hr.ras',
#             # 'RSM':RSM('NGO', 'LNO', (-2,0,3)),
#             # 'label':'C1-02607'}
# # # windows = {  'RSM_file':"C1-02609_RSM 2_5deg_t_2deg w_13hr_scanning1D.xrdml",
# # #             'RSM':RSM('NGO', 'LNO', (1,1,3)),
# # #             'label':'C1-02609'}
            
# # ##Import
# rsm1['RSM'].importData(rsm1['RSM_file'])
# # rsm2['RSM'].importData(rsm2['RSM_file'])
# # # rsm3['RSM'].importData(rsm3['RSM_file'])
# # # # rsm4['RSM'].importData(rsm4['RSM_file'])

# # # threshmax = rsm1['RSM'].intensity.max()
# # # threshmin = rsm1['RSM'].intensity.min()
# # print(rsm2['RSM'].intensity.max())

# # # # # ##plot
# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(6,6))
# rsm1['RSM'].plot(fig=fig,ax=ax1,show=False,colormap='gist_heat_r', threshImin=500*rsm1['RSM'].intensity.min(),xmin=-2.32, xmax=-2.25, ymin=4.8,ymax=5.1)
# # rsm2['RSM'].plot(fig=fig,ax=ax2,show=False,colormap = 'gist_heat_r', threshImin=500*rsm2['RSM'].intensity.min(),xmin=-2.32, xmax=-2.25, ymin=4.8,ymax=5.1)
# # # # # rsm3['RSM'].plot(fig=fig,ax=ax3,show=False,grid=True,threshImax=threshmax)
# # # ax1.set_title('C1-02653 (5 nm)')
# # # ax2.set_title('C1-02654 (10 nm)')
# # ax1.grid(visible=True)
# # ax1.grid(visible=True)
# # # # # ax3.set_title('3. (0-26)')
# # # # # rsm4['RSM'].plot(fig=fig,ax=ax4,show=False,grid=True)


# fig.tight_layout()
# plt.show()
# # plt.savefig("RSM_C1-2590_026.png",format='png',dpi=400)

# # qLNO = ttq(46.9)q2a
# # qNNO = ttq(47.22)
# # print(qLNO,qNNO)

# # plt = plotSLACxrd('/Users/oschn/Dropbox/SSRL/2-1/C3_02440_scans/C3_02440_001',Q=True)


# # path = '/Users/oschnebl/Library/CloudStorage/Box-Box/RNO Neuromorphic LDRD/Data/C1/02607/RSM/*/*2t-w*.ras'
# # files=glob.glob(path)
# # plotXRD(files,together=True,legend=False)

