import numpy as np
import matplotlib.pyplot as plt

fileNum = 1
for i in range(1, fileNum + 1):
    if i < 41:
        if i < 10:
            locals()['a' + str(i)] = np.loadtxt(r'D:\1806\data_aligned\C30' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 17:
            locals()['a' + str(i)] = np.loadtxt(r'D:\1806\data_aligned\C3' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 26:
            locals()['a' + str(i)] = np.loadtxt(
                r'D:\1806\data_aligned\C50' + str(i - 16) + '.txt', skiprows=0)
        else:
            locals()['a' + str(i)] = np.loadtxt(r'D:\1806\data_aligned\C5' + str(i - 16) + '.txt',
                                                skiprows=0)

gap = 20
which_data = 7
data = a1[:, which_data]
data_gap = a1[0:-1:gap, which_data]
time = np.arange(len(data))/100*np.sqrt(56)
time_gap = time[0:-1:gap]

plt_len = 2500
begin_loc = int(180*100/np.sqrt(56))
plt.plot(time[begin_loc:plt_len-gap+begin_loc], data[begin_loc:plt_len-gap+begin_loc], 'r', label='Original data')
plt.plot(time[-plt_len:]-time[-plt_len]+600, data[-plt_len:], 'r')
plt.plot(time_gap[begin_loc//gap:plt_len//gap+begin_loc//gap], data_gap[begin_loc//gap:plt_len//gap+begin_loc//gap], 'b', label='Data processed by downsampling')
plt.plot(time_gap[-plt_len//gap:]-time[-plt_len]+600, data_gap[-plt_len//gap:], 'b')
# plt.plot(time[plt_len-gap+begin_loc+100:int(600*100/np.sqrt(56))-100], [data[plt_len-gap+begin_loc]]*len(time[plt_len-gap+begin_loc+100:int(600*100/np.sqrt(56))-100]), '--')
plt.plot(time[plt_len-gap+begin_loc+100:int(600*100/np.sqrt(56))-100], [0]*len(time[plt_len-gap+begin_loc+100:int(600*100/np.sqrt(56))-100]), '--')

scale_ls = [200, 300, 400, 500, 600, 700, 800]
index_ls = ['200', '300', '', '' ,'169900' ,'170000' ,'170100']
_ = plt.xticks(scale_ls, index_ls, fontsize=20)
_ = plt.yticks(fontsize=20)

if which_data == 1:
    plt.ylabel(r'$\eta(m)$', fontsize=24)
elif which_data == 2:
    plt.ylabel(r'$surge(m)$', fontsize=24)
elif which_data == 3:
    plt.ylabel(r'$sway(m)$', fontsize=24)
elif which_data == 4:
    plt.ylabel(r'$heave(m)$', fontsize=24)
elif which_data == 5:
    plt.ylabel(r'$roll(deg)$', fontsize=24)
elif which_data == 6:
    plt.ylabel(r'$pitch(deg)$', fontsize=24)
elif which_data == 7:
    plt.ylabel(r'$yaw(deg)$', fontsize=24)

plt.xlabel('time(s)', fontsize=24)

font1 = {'family':'Times New Roman',
         'weight':'normal',
         'size':24,}
plt.legend(prop=font1, loc="upper left")  #set legend location

plt.show()



# which_data = 2
# fig = plt.figure(figsize=(6,18))
# for which_data in range(1, 8):
#     locals()['ax'+str(which_data)] = fig.add_subplot(7,1,which_data)
#     data = a1[:, which_data]
#     data_gap = a1[0:-1:gap, which_data]
#     time = np.arange(len(data))/100*np.sqrt(56)
#     time_gap = time[0:-1:gap]
#
#     plt_len = 2500
#     begin_loc = int(180*100/np.sqrt(56))
#     locals()['ax' + str(which_data)].plot(time[begin_loc:plt_len-gap+begin_loc], data[begin_loc:plt_len-gap+begin_loc], 'r', label='Original data')
#     locals()['ax' + str(which_data)].plot(time[-plt_len:]-time[-plt_len]+600, data[-plt_len:], 'r')
#     locals()['ax' + str(which_data)].plot(time_gap[begin_loc//gap:plt_len//gap+begin_loc//gap], data_gap[begin_loc//gap:plt_len//gap+begin_loc//gap], 'b', label='Data processed by downsampling')
#     locals()['ax' + str(which_data)].plot(time_gap[-plt_len//gap:]-time[-plt_len]+600, data_gap[-plt_len//gap:], 'b')
#     locals()['ax' + str(which_data)].plot(time[plt_len-gap+begin_loc+100:int(600*100/np.sqrt(56))-100], [0]*len(time[plt_len-gap+begin_loc+100:int(600*100/np.sqrt(56))-100]), '--')
#     if which_data == 7:
#         plt.xlabel('time(s)', fontsize=6)
#
#         scale_ls = [200, 300, 400, 500, 600, 700, 800]
#         index_ls = ['200', '300', '', '' ,'169900' ,'170000' ,'170100']
#         _ = plt.xticks(scale_ls, index_ls)
#     else:
#         locals()['ax' + str(which_data)].set_xticks([])
#
#     plt.ylabel(r'$\eta(cm)$', fontsize=6)
#     font1 = {'family':'Times New Roman',
#              'weight':'normal',
#              'size':6,}
#     if which_data == 1:
#         plt.legend(prop=font1, loc="upper left")  #set legend location
#
# # locals()['ax' + str(which_data)].axis["bottom"].set_visible(False)
# plt.show()
