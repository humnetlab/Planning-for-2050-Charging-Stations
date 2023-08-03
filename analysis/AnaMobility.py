import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

fsize = 12; tdir = 'in'; major = 6; minor = 3; lwidth = 1; lhandle = 1
space_symbol = '                          '
mymarkersize  = 4
color_platte = ['#00429d', '#4771b2', '#73a2c6', '#a5d5d8', '#ffffe0', '#fcc17e', '#e0884e', '#bc4f3c', '#93003a']
sns.set_theme(style="white")
plt.rcParams['font.size'] = fsize
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['legend.fontsize'] = fsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.labelsize'] = fsize
plt.rcParams['ytick.labelsize'] = fsize
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
plt.rcParams['axes.linewidth'] = lwidth
plt.rcParams['legend.handlelength'] = lhandle
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['xtick.bottom'] = True
plt.rcParams.update({'font.family':'Arial'})
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_platte)


class AnaMobility():
    def __init__(self, work_l2_access,home_l2_access):
        super().__init__()
        self.work_l2_access = work_l2_access
        self.home_l2_access = home_l2_access
        self.mobility_folder_name = os.path.join('..','result','mobility')
        self.adopter_folder_name = os.path.join('..','result','adopter','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access))
        self.figure_folder_name = os.path.join('..','figure','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access))

    def visMobility(self):
        id_list = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(0) + 'p.pkl'), 'rb'), encoding='bytes')
        userTraj_label = pickle.load(open(os.path.join(self.mobility_folder_name,'userTraj_label_week.pkl'), 'rb'), encoding='bytes')
        userTraj_zipcode = pickle.load(open(os.path.join(self.mobility_folder_name,'userTraj_zipcode_week.pkl'), 'rb'), encoding='bytes')
        userTraj_time = pickle.load(open(os.path.join(self.mobility_folder_name,'userTraj_time_week.pkl'), 'rb'), encoding='bytes')
        userTraj_energy = pickle.load(open(os.path.join(self.mobility_folder_name,'userTraj_renergy_week.pkl'), 'rb'), encoding='bytes')

        user_id = []; arrive_time = []; depature_time = []; stay_type = []; stay_zipcode = []
        for id in id_list:
            for i in range(len(userTraj_label[id])):
                user_id.append(id)
                stay_type.append(userTraj_label[id][i])
                stay_zipcode.append(userTraj_zipcode[id][i])
                try:
                    dtime = userTraj_time[id][i+1]
                    depature_time.append(userTraj_time[id][i+1]%144+1)
                except:
                    dtime = 144
                    depature_time.append(144)
                if i>0:
                    arrive_time.append(min(userTraj_time[id][i],dtime)%144+1)
                else:
                    arrive_time.append(1)

        df_trip_energy = pd.DataFrame(data={'id': user_id, 'arrive_time': arrive_time, 'depature_time':depature_time, 'stay_type':stay_type, 'stay_zipcode':stay_zipcode})

        table = pd.pivot_table(df_trip_energy, values='stay_type', index=['id'],aggfunc=np.count_nonzero)
        fig, axs = plt.subplots(nrows=2, ncols=2, sharey=False, figsize=(6,4))
        bin = 13
        i = 0; j = 0
        x= df_trip_energy[df_trip_energy['arrive_time']!=1]['arrive_time']
        axs[i,j].hist(df_trip_energy[df_trip_energy['arrive_time']!=1]['arrive_time'], bin, weights = [1./len(x)]*len(x),  edgecolor='k', facecolor=color_platte[1])
        axs[i,j].set_xlabel('Arrive time [10 min]')
        axs[i,j].set_ylabel('Frequency')
        axs[i,j].set_title('Arrive time')
        axs[i,j].set_ylim(0,0.2)
        axs[i,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        i = 0; j = 1
        x= df_trip_energy[df_trip_energy['depature_time']!=144]['depature_time']
        axs[i,j].hist(df_trip_energy[df_trip_energy['depature_time']!=144]['depature_time'], bin, weights = [1./len(x)]*len(x), edgecolor='k',facecolor=color_platte[3])
        axs[i,j].set_xlabel('Depature time [10 min]')
        axs[i,j].set_ylabel('Frequency')
        axs[i,j].set_title('Depature time')
        axs[i,j].set_ylim(0,0.2)
        axs[i,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        i = 1; j = 0
        x = pd.DataFrame.from_dict(userTraj_energy, orient='index')/7
        axs[i,j].hist(x, bin, weights = [1./len(x)]*len(x), edgecolor='k',facecolor=color_platte[5])
        axs[i,j].set_xlabel('Daily Energy Consumption [kWh]')
        axs[i,j].set_ylabel('Frequency')
        axs[i,j].set_title('Daily Energy Consumption')
        axs[i,j].set_ylim(0,0.6)
        axs[i,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        i = 1; j = 1
        x = table['stay_type'].values/7
        axs[i,j].hist(x, bin, weights = [1./len(x)]*len(x), edgecolor='k', facecolor=color_platte[7])
        axs[i,j].set_xlabel('Number of Stay')
        axs[i,j].set_ylabel('Frequency')
        axs[i,j].set_title('Number of Stay')
        axs[i,j].set_ylim(0,0.6)
        axs[i,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        if not os.path.exists(self.figure_folder_name):
            os.makedirs(self.figure_folder_name)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_folder_name,'sp_overview.pdf'), dpi = 900, pad_inches = .1, bbox_inches = 'tight')
        plt.show()

if __name__ == "__main__":
    anamb = AnaMobility(0.50,0.54)
    anamb.visMobility()


