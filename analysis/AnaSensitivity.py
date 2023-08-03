import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import itertools
import os
import warnings
warnings.filterwarnings("ignore")

class AnaSensitivity():
    def __init__(self, work_l2_access_list,home_l2_access_list,peak_start_list,acceptance_list,max_stay_list):
        super().__init__()
        self.work_l2_access = work_l2_access_list
        self.home_l2_access = home_l2_access_list
        self.peak_start = peak_start_list
        self.acceptance = acceptance_list
        self.max_stay = max_stay_list
        self.figure_folder_name = os.path.join('..','figure','sensitivity')
        if not os.path.exists(self.figure_folder_name):
            os.makedirs(self.figure_folder_name)
    
    def visSensitivity(self,fname,findex):
        peakridges = pd.DataFrame([]); offpeakridges = pd.DataFrame([])
        for item in itertools.product(self.work_l2_access, self.home_l2_access,self.peak_start,self.acceptance,self.max_stay):
            work_l2_access,home_l2_access,peak_start,acceptance,max_stay = item
            folder_name = os.path.join('..','result','shift','work_'+str(work_l2_access)+'home_'+str(home_l2_access)+'peak_'+str(peak_start)+'_'+str(peak_start+3)+'acceptance_'+str(acceptance)+'max_stay_'+str(max_stay))
            demand_supply = pd.read_csv(os.path.join(folder_name,'demand.csv'))
            demand_supply_results = demand_supply[demand_supply['adoption rate']>3]

            peakridge = demand_supply_results.copy()
            peakridge[fname] = item[findex]
            peakridge['day'] = 'Grid Peak Hours'
            peakridge['peak value'] = (peakridge['week_peak_before']-peakridge['week_peak_after'])/peakridge['week_peak_before']
            peakridges = pd.concat([peakridges,peakridge.copy()])

            offpeakridge = demand_supply_results.copy()
            offpeakridge[fname] = item[findex]
            offpeakridge['day'] = 'Grid Off-Peak Hours'
            offpeakridge['peak value'] = (offpeakridge['week_offpeak_after']-offpeakridge['week_offpeak_before'])/offpeakridge['week_offpeak_before']
            offpeakridges = pd.concat([offpeakridges,offpeakridge.copy()])

        g = sns.catplot(
            data=peakridges, kind='bar',
            x='adoption rate', y='peak value', hue=fname, hatch='.',
            errorbar="sd", capsize=.05, height=4, palette = sns.color_palette(['#00429d', '#73a2c6', '#ffffe0', '#e0884e'])
        )
        g.fig.set_figwidth(10)
        g.fig.set_figheight(3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(os.path.join(self.figure_folder_name,'sp_sensi_'+fname+'.pdf'), dpi = 300, pad_inches = .1, bbox_inches = 'tight')
        plt.show()

if __name__ == "__main__":
    work_l2_access_list = [0.5]; home_l2_access_list = [0.541]
    peak_start_list = [17]
    acceptance_list = [1]; max_stay_list = [3,5,7,9]
    anasen = AnaSensitivity(work_l2_access_list,home_l2_access_list,peak_start_list,acceptance_list,max_stay_list)
    anasen.visSensitivity('maxstay',4)
    