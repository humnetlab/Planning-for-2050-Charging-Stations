import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as cx
import os
import warnings
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as clr
from geopandas import GeoDataFrame
import plotly.graph_objects as go 
warnings.filterwarnings("ignore")

fsize = 12; tdir = 'in'; major = 6; minor = 4; lwidth = 1; lhandle = 1
space_symbol = '                          '
color_platte = ['#00429d', '#4771b2', 
'#73a2c6', '#a5d5d8', '#ffffe0', '#fcc17e', '#e0884e', '#bc4f3c', '#93003a']
plt.style.use('default')
sns.set_theme(style="white")
plt.rcParams['figure.figsize'] = [6.4, 4.8]
plt.rcParams['font.size'] = fsize
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

colors= ['#001144', '#00245a', '#013871', '#0d4c89', '#305f9e', '#4973b4', '#6087ca', '#779ce1', '#8eb2f9', '#a7caff', '#c1e4ff'][::-1]
nodes = np.arange(0,1.1,0.1)
homecmap = clr.LinearSegmentedColormap.from_list("homecmap", list(zip(nodes, colors)))
colors= ['#001819', '#012b2e', '#0a3e41', '#235255', '#386669', '#4d7b7e', '#639194', '#78a7aa', '#8fbec1', '#a5d5d8', '#bdedf0'][::-1]
nodes = np.arange(0,1.1,0.1)
workcmap = clr.LinearSegmentedColormap.from_list("workcmap", list(zip(nodes, colors)))
colors= ['#310300', '#401a00', '#532d00', '#694001', '#805313', '#976728', '#ae7b3c', '#c69050', '#dea664', '#f7bc79', '#ffd995'][::-1]
nodes = np.arange(0,1.1,0.1)
othercmap = clr.LinearSegmentedColormap.from_list("othercmap", list(zip(nodes, colors)))
colors= ['#440000', '#580000', '#700401', '#892216', '#a23929', '#bc4f3c', '#d66550', '#f17c65', '#ff937b', '#ffab91', '#ffc3a8'][::-1]
nodes = np.arange(0,1.1,0.1)
totalcmap = clr.LinearSegmentedColormap.from_list("totalcmap", list(zip(nodes, colors)))
colors_map = [homecmap,workcmap,othercmap,totalcmap]
colors= ['#001144', '#00245a', '#013871', '#0d4c89', '#305f9e', '#4973b4', '#6087ca', '#779ce1', '#8eb2f9', '#a7caff', '#c1e4ff']
nodes = np.arange(0,1.1,0.1)
bluecmap = clr.LinearSegmentedColormap.from_list("bluecmap", list(zip(nodes, colors)))
colors= ['#440000', '#580000', '#700401', '#892216', '#a23929', '#bc4f3c', '#d66550', '#f17c65', '#ff937b', '#ffab91', '#ffc3a8']
nodes = np.arange(0,1.1,0.1)
redcmap = clr.LinearSegmentedColormap.from_list("redcmap", list(zip(nodes, colors)))


class AnaShift():
    def __init__(self, work_l2_access,home_l2_access,peak_start,peak_end,acceptance,max_stay):
        super().__init__()
        self.work_l2_access = work_l2_access
        self.home_l2_access = home_l2_access
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.acceptance = acceptance
        self.max_stay = max_stay
        self.mobility_folder_name = os.path.join('..','result','mobility')
        self.adopter_folder_name = os.path.join('..','result','adopter','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access))
        self.behavior_folder_name = os.path.join('..','result','behavior','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access))
        self.shift_folder_name = os.path.join('..','result','shift','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access)+'peak_'+str(self.peak_start)+'_'+str(self.peak_end)+'acceptance_'+str(self.acceptance)+'max_stay_'+str(self.max_stay))
        self.figure_folder_name = os.path.join('..','figure','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access)+'peak_'+str(self.peak_start)+'_'+str(self.peak_end)+'acceptance_'+str(self.acceptance)+'max_stay_'+str(self.max_stay))
        self.df_user_session = pd.read_csv(os.path.join(self.behavior_folder_name,'simulated_session.csv'))
        self.df_shift_session = pd.read_csv(os.path.join(self.shift_folder_name,'shifted_session.csv')) 
        self.demand = pd.read_csv(os.path.join(self.shift_folder_name,'demand.csv'))
        self.supply = pd.read_csv(os.path.join(self.shift_folder_name,'supply.csv'))
        self.geozip = gpd.read_file(os.path.join('..','data','census','sfbay_zip.geojson')) 
        self.geocounty = gpd.read_file(os.path.join('..','data','census','Bay Area Counties.geojson')) 
        self.user_select = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(0) + 'p.pkl'), 'rb'), encoding='bytes')
        if not os.path.exists(self.figure_folder_name):
            os.makedirs(self.figure_folder_name)

    def simEVLoad(self,df):
        zipcode_list = self.df_user_session['stay_zipcode'].unique().copy()

        home_l2_rate = 6.6;work_l2_rate = 6.6;mud_l2_rate = 6.6;public_l2_rate = 6.6;home_l1_rate = 1.2;public_l3_rate = 50;dayhour = 24*7

        df_home_l1 = df[(df['session_type']=='home_l1')]
        df_home_l1['charge_end_time'] = np.minimum(df_home_l1['arrive_time']+(df_home_l1['session_energy']/home_l1_rate)*6,df_home_l1['depature_time'])

        df_home_l2 = df[(df['session_type']=='home_l2')]
        df_home_l2['charge_end_time'] = np.minimum(df_home_l2['arrive_time']+(df_home_l2['session_energy']/home_l2_rate)*6,df_home_l2['depature_time'])
        
        df_home_l3 = df[(df['session_type']=='mud_l2')]
        df_home_l3['charge_end_time'] = np.minimum(df_home_l3['arrive_time']+(df_home_l3['session_energy']/mud_l2_rate)*6,df_home_l3['depature_time'])
        
        df_work = df[df['session_type']=='work_l2']
        df_work['charge_end_time'] = np.minimum(df_work['arrive_time']+(df_work['session_energy']/work_l2_rate)*6,df_work['depature_time'])
        
        df_other_l2 = df[df['session_type']=='public_l2']
        df_other_l2['charge_end_time'] = np.minimum(df_other_l2['arrive_time']+(df_other_l2['session_energy']/public_l2_rate)*6,df_other_l2['depature_time'])
        df_other_l3 = df[df['session_type']=='public_l3']
        df_other_l3['charge_end_time'] = np.minimum(df_other_l3['arrive_time']+(df_other_l3['session_energy']/public_l3_rate)*6,df_other_l3['depature_time'])

        home_demand_l1 = {}; home_demand_l2 = {}; home_demand_l3 = {}; other_demand_l2 = {}; other_demand_l3 = {}; 
        total_demand = {}; home_demand = {}; other_demand = {}; work_demand = {}

        for zipcode in zipcode_list:
            df_home_1 = df_home_l1[(df_home_l1['session_energy']!=0)&(df_home_l1['stay_zipcode']==zipcode)]
            start_time_1 = (df_home_1['arrive_time']/6).astype(int).values
            end_time_1 = (df_home_1['charge_end_time']/6).astype(int).values
            home_matrix = np.zeros((len(df_home_1),dayhour))
            for i in range(len(start_time_1)):
                home_matrix[i,start_time_1[i]:end_time_1[i]+1] = home_l1_rate
            home_demand_l1[zipcode] = np.sum(home_matrix,axis=0)

            df_home_1 = df_home_l2[(df_home_l2['session_energy']!=0)&(df_home_l2['stay_zipcode']==zipcode)]
            start_time_1 = (df_home_1['arrive_time']/6).astype(int).values
            end_time_1 = (df_home_1['charge_end_time']/6).astype(int).values
            home_matrix = np.zeros((len(df_home_1),dayhour))
            for i in range(len(start_time_1)):
                home_matrix[i,start_time_1[i]:end_time_1[i]+1] = home_l2_rate
            home_demand_l2[zipcode] = np.sum(home_matrix,axis=0)

            df_home_1 = df_home_l3[(df_home_l3['session_energy']!=0)&(df_home_l3['stay_zipcode']==zipcode)]
            start_time_1 = (df_home_1['arrive_time']/6).astype(int).values
            end_time_1 = (df_home_1['charge_end_time']/6).astype(int).values
            home_matrix = np.zeros((len(df_home_1),dayhour))
            for i in range(len(start_time_1)):
                home_matrix[i,start_time_1[i]:end_time_1[i]+1] = mud_l2_rate
            home_demand_l3[zipcode] = np.sum(home_matrix,axis=0)

            df_work_1 = df_work[(df_work['session_energy']!=0)&(df_work['stay_zipcode']==zipcode)]
            start_time =(df_work_1['arrive_time']/6).astype(int).values
            end_time = (df_work_1['charge_end_time']/6).astype(int).values
            work_matrix = np.zeros((len(df_work_1),dayhour))
            for i in range(len(start_time)):
                work_matrix[i,start_time[i]:end_time[i]+1] = work_l2_rate
            work_demand[zipcode] = np.sum(work_matrix,axis=0)

            df_other_1 = df_other_l2[(df_other_l2['session_energy']!=0)&(df_other_l2['stay_zipcode']==zipcode)]
            start_time =(df_other_1['arrive_time']/6).astype(int).values
            end_time = (df_other_1['charge_end_time']/6).astype(int).values
            other_matrix = np.zeros((len(df_other_1),dayhour))
            for i in range(len(start_time)):
                other_matrix[i,start_time[i]:end_time[i]+1] = public_l2_rate
            other_demand_l2[zipcode] = np.sum(other_matrix,axis=0)

            df_other_1 = df_other_l3[(df_other_l3['session_energy']!=0)&(df_other_l3['stay_zipcode']==zipcode)]
            start_time =(df_other_1['arrive_time']/6).astype(int).values
            end_time = (df_other_1['charge_end_time']/6).astype(int).values
            other_matrix = np.zeros((len(df_other_1),dayhour))
            for i in range(len(start_time)):
                other_matrix[i,start_time[i]:end_time[i]+1] = public_l3_rate
            other_demand_l3[zipcode] = np.sum(other_matrix,axis=0)

            home_demand[zipcode] = home_demand_l1[zipcode]+home_demand_l2[zipcode]+home_demand_l3[zipcode]
            other_demand[zipcode] = other_demand_l3[zipcode]+other_demand_l2[zipcode]
            total_demand[zipcode] = work_demand[zipcode]+other_demand[zipcode]+home_demand[zipcode]
        
        home = pd.DataFrame.from_dict(home_demand, orient='index')
        work = pd.DataFrame.from_dict(work_demand, orient='index')
        other = pd.DataFrame.from_dict(other_demand, orient='index')
        total = pd.DataFrame.from_dict(total_demand, orient='index')
        return home, work, other, total

    def visStation(self):
        info = pd.read_csv(os.path.join('..','data','supply','usage_type.csv'))
        geocs = gpd.read_file(os.path.join('..','data','supply','evcs.geojson'))
         
        geocs['UsageTypeID'] = geocs['UsageTypeID'].astype(int)
        geocs = geocs.merge(info,left_on='UsageTypeID',right_on='ID')
        evcs_sfbay = gpd.sjoin(self.geozip,geocs,how="inner").copy()
        evcs_sfbay['Power'] = evcs_sfbay['PowerKW'] * evcs_sfbay['Quantity']
        evcs_sfbay = evcs_sfbay[evcs_sfbay['StationType']=='Public Charging Places']

        evcs_sfbay_vis = pd.pivot_table(evcs_sfbay, values='Quantity',columns=['LevelID'], index=['ZCTA5CE10'],aggfunc=np.sum)
        evcs_sfbay_vis = np.sum(evcs_sfbay_vis,axis=0)
        evcs_sfbay_vis = evcs_sfbay_vis.rename(index={1: "Level 1", 2: "Level 2", 3: "Level 3"})

        fig, ax = plt.subplots(figsize=(3,3))
        wedges, texts, autotexts = ax.pie(evcs_sfbay_vis.values,colors=[ '#e0884e','#fcc17e', '#bc4f3c'], explode=(0.1,0.1,0.1),autopct=' ',textprops=dict(color="k"))

        ax.legend(wedges, evcs_sfbay_vis.index,
                  loc="center left", frameon=False,
                  bbox_to_anchor=(1, 0, 0.5, 1),fontsize=15)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_folder_name,'fig3_pie_station.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        evcs_sfbay_vis = pd.pivot_table(evcs_sfbay, values='Quantity', index=['ZCTA5CE10'],aggfunc=np.sum).reset_index()
        geozip_merge = self.geozip.merge(evcs_sfbay_vis,right_on='ZCTA5CE10',left_on='ZCTA5CE10', how='left').copy()
        geozip_merge = gpd.sjoin(geozip_merge,self.geocounty)
        geozip_merge = pd.pivot_table(geozip_merge, values=['Quantity'], index=['county'], aggfunc=np.sum).reset_index()
        geozip_merge = GeoDataFrame(geozip_merge.merge(self.geocounty,right_on='county',left_on='county', how='left'))

        fig, ax = plt.subplots(figsize=(8,3))
        ax= geozip_merge.plot(ax = ax, column ='Quantity',cmap='Blues', legend=False, edgecolor='grey')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-126, -120.5)
        ax.set_ylim(36.7, 39.0)
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, crs=geocs.crs, attribution_size=6)
        cb = plt.colorbar(ax.collections[0], ax=ax, shrink=0.6, label = space_symbol, location = 'right')
        cb.outline.set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_folder_name,'fig3_geo_station.pdf'),format='pdf',dpi=300,transparent=True)
        plt.show()

    def visTempLoad(self):
        df_before = self.df_user_session[self.df_user_session['id'].isin(self.user_select)].copy()
        df_after = self.df_shift_session[self.df_shift_session['id'].isin(self.user_select)].copy()
        df_before = df_before.fillna(-1); df_after = df_after.fillna(-1)
        df_before = pd.pivot_table(df_before,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()
        df_after = pd.pivot_table(df_after,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()
        home_load, work_load, other_load, total_load = self.simEVLoad(df_before)
        homes_load, works_load, others_load, totals_load = self.simEVLoad(df_after)

        peak_start = 17; peak_end=21
        week_peak = np.zeros(168); week_offpeak = np.zeros(168)
        for hour in range(168):
            if hour<24*5:
                if (hour%24 > peak_start) and (hour%24 < peak_end):
                    week_peak[hour] = 1
                elif (hour%24 == peak_start) or (hour%24 == peak_end):
                    week_peak[hour] = 1
                    week_offpeak[hour] = 1
                else:
                    week_offpeak[hour] = 1
            else:
                week_offpeak[hour] = 1        

        fig, ax = plt.subplots(nrows=2, ncols=2, sharey = True, figsize=(8,3.5))
        legend_elements = [Patch(facecolor=color_platte[7], alpha=0.5,
                                label='                          '),
                        Patch(facecolor='lightgrey', alpha=0.5,
                                label='                          ')]
        fig.legend(handles=legend_elements,frameon=False,ncol=2,bbox_to_anchor=(0.75, 1.05))
        ax[0,0].plot(np.sum(home_load/1000,axis=0),c ='darkgrey',label='         ',linewidth=3)
        ax[0,0].plot(np.sum(homes_load/1000,axis=0),c =color_platte[1],label='         ',linestyle = 'dotted',linewidth=3)
        ax[0,0].fill_between(range(168), 0, 1, where=week_peak, alpha=0.2,  linewidth =0, color=color_platte[7], transform=ax[0,0].get_xaxis_transform())
        ax[0,0].fill_between(range(168), 0, 1, where=week_offpeak, alpha=0.3, linewidth =0,  color='lightgrey', transform=ax[0,0].get_xaxis_transform())
        ax[0,0].set_xticks([])
        ax[0,0].legend(frameon=False)

        ax[0,1].plot(np.sum(work_load/1000,axis=0),c ='darkgrey',label='         ',linewidth=3)
        ax[0,1].plot(np.sum(works_load/1000,axis=0),c =color_platte[3],label='         ',linestyle = 'dotted',linewidth=3)
        ax[0,1].fill_between(range(168), 0, 1, where=week_peak, alpha=0.2,  linewidth =0, color=color_platte[7], transform=ax[0,1].get_xaxis_transform())
        ax[0,1].fill_between(range(168), 0, 1, where=week_offpeak, alpha=0.3, linewidth =0,  color='lightgrey', transform=ax[0,1].get_xaxis_transform())
        ax[0,1].set_xticks([])
        ax[0,1].legend(frameon=False)

        ax[1,0].plot(np.sum(other_load/1000,axis=0),c = 'darkgrey', label='         ',linewidth=3)
        ax[1,0].plot(np.sum(others_load/1000,axis=0),c =color_platte[5],label='         ',linestyle = 'dotted',linewidth=3)
        ax[1,0].fill_between(range(168), 0, 1, where=week_peak, alpha=0.2,  linewidth =0, color=color_platte[7], transform=ax[1,0].get_xaxis_transform())
        ax[1,0].fill_between(range(168), 0, 1, where=week_offpeak, alpha=0.3, linewidth =0,  color='lightgrey', transform=ax[1,0].get_xaxis_transform())
        ax[1,0].set_xticks([0,24,48,72,96,120,144,168])
        ax[1,0].legend(frameon=False)

        ax[1,1].plot(np.sum(total_load/1000,axis=0),c = 'darkgrey',label='         ',linewidth=3)
        ax[1,1].plot(np.sum(totals_load/1000,axis=0),c =color_platte[7],label='         ',linestyle = 'dotted',linewidth=3)
        ax[1,1].fill_between(range(168), 0, 1, where=week_peak, alpha=0.2,  linewidth =0, color=color_platte[7], transform=ax[1,1].get_xaxis_transform())
        ax[1,1].fill_between(range(168), 0, 1, where=week_offpeak, alpha=0.3, linewidth =0,  color='lightgrey', transform=ax[1,1].get_xaxis_transform())
        ax[1,1].set_xticks([0,24,48,72,96,120,144,168])
        ax[1,1].legend(frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.savefig(os.path.join(self.figure_folder_name,'fig3_temporal_shifting.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()
        
    def visShaveSector(self): 
        
        df_shift_session = self.df_shift_session[self.df_shift_session['id'].isin(self.user_select)].copy()
        df_shift_total = df_shift_session[df_shift_session['is_shift']>=1].copy()

        df_shift_home = df_shift_total[df_shift_total['original_session_type'].isin(['home_l1','home_l2','mud_l2'])]
        home_home = len(df_shift_home[df_shift_home['session_type'].isin(['home_l1','home_l2','mud_l2'])])
        home_work = len(df_shift_home[df_shift_home['session_type'].isin(['work_l2'])])
        home_public = len(df_shift_home[df_shift_home['session_type'].isin(['public_l2','public_l3'])])
        home_sum = len(df_shift_home)

        df_shift_work = df_shift_total[df_shift_total['original_session_type'].isin(['work_l2'])]
        work_work = len(df_shift_work[df_shift_work['session_type'].isin(['work_l2'])])
        work_home = len(df_shift_work[df_shift_work['session_type'].isin(['home_l1','home_l2','mud_l2'])])
        work_public = len(df_shift_work[df_shift_work['session_type'].isin(['public_l2','public_l3'])])
        work_sum = len(df_shift_work)

        df_shift_public = df_shift_total[df_shift_total['original_session_type'].isin(['public_l2','public_l3'])]
        public_public = len(df_shift_public[df_shift_public['session_type'].isin(['public_l2','public_l3'])])
        public_work = len(df_shift_public[df_shift_public['session_type'].isin(['work_l2'])])
        public_home = len(df_shift_public[df_shift_public['session_type'].isin(['home_l1','home_l2','mud_l2'])])
        public_sum = len(df_shift_public)
        
        node_label = ['Home: Before', 'Work: Before', 'Public: Before', 'Home: After', 'Work: After', 'Public: After']
        node_dict = {y:x for x, y in enumerate(node_label)}
        source = ['Home: Before','Home: Before','Home: Before','Work: Before','Work: Before','Work: Before','Public: Before','Public: Before','Public: Before']
        target = ['Home: After','Work: After','Public: After','Home: After','Work: After','Public: After','Home: After','Work: After','Public: After'] 
        values = [home_home/home_sum, home_work/home_sum, home_public/home_sum, work_home/work_sum, work_work/work_sum, work_public/work_sum, public_home/public_sum, public_work/public_sum, public_public/public_sum]
        
        source_node = [node_dict[x] for x in source]
        target_node = [node_dict[x] for x in target]

        fig = go.Figure( 
            data=[go.Sankey(
                arrangement = "snap",
                node = dict( 
                    label = node_label,
                    x = [0.1,0.1,0.1,0.7,0.7,0.7],
                    y = [0.1,0.2,0.3,0.1,0.2,0.3]
                ),
                link = dict(
                    source = source_node,
                    target = target_node,
                    value = values
                ))])
        fig.write_image(os.path.join(self.figure_folder_name,'fig3_sanky.pdf'))

    def visZipScatter(self):
        current_rate = 2
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        demand_supply = self.demand.copy()

        demand_supply_results = demand_supply[demand_supply['adoption rate']==current_rate]
        g = sns.jointplot(x='week_peak_before', y='week_peak_after', data=demand_supply_results, 
                        kind="reg", truncate=True,height=3.5,xlim=(0,1000),ylim=(0,1000),color=color_platte[1], scatter_kws={"s": 8})
        g.ax_joint.set_xlabel('          ',fontsize=fsize)
        g.ax_joint.set_ylabel('          ',fontsize=fsize)
        plt.plot([0, 3000], [0, 3000], ls="--", c=".3", label='        ')
        plt.legend(frameon=False, fontsize = fsize, loc = [0.0,0.8])
        plt.savefig(os.path.join(self.figure_folder_name,'fig3_joint_peak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        g = sns.jointplot(x='week_offpeak_before', y='week_offpeak_after', data=demand_supply_results,
                        kind="reg", truncate=True,height=3.5,xlim=(0,2000),ylim=(0,2000),color=color_platte[7], scatter_kws={"s": 8})
        plt.plot([0, 3000], [0, 3000], ls="--", c=".3", label='           ')
        plt.legend(frameon=False, fontsize = fsize, loc = [0.35,0.1])
        g.ax_joint.set_xlabel('          ',fontsize=fsize)
        g.ax_joint.set_ylabel('          ',fontsize=fsize)
        plt.savefig(os.path.join(self.figure_folder_name,'fig3_joint_offpeak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()
    
    def visZIPPair(self):
        demand_supply = self.supply.copy()
        
        demand_supply['zipcode'] = demand_supply['zipcode'].astype(str)
        demand_supply_results = demand_supply[demand_supply['adoption rate']==2]
        geozip_merge = self.geozip.merge(demand_supply_results,right_on='zipcode',left_on='ZCTA5CE10').copy()
        geocounty = gpd.sjoin(geozip_merge,self.geocounty).copy()
        contyheat = pd.pivot_table(geocounty, values=['week_peak_before', 'week_offpeak_before', 'week_peak_after',
            'week_offpeak_after','Power'], index=['zipcode','county'], aggfunc=np.sum).reset_index()
        geocounty = geocounty[['zipcode', 'week_peak_before', 'week_offpeak_before',
            'week_peak_after', 'week_offpeak_after', 'Power', 'adoption rate','county']]

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        countyvis = pd.melt(contyheat, id_vars=['zipcode','Power','county'], value_vars=['week_peak_before', 'week_peak_after'])
        X_coords = np.array([contyheat.week_peak_before.values,contyheat.week_peak_after.values])
        Y_coords = np.array([contyheat.Power.values,contyheat.Power.values])
        plt.figure(figsize=(3,6.5))

        plt.plot(X_coords, 
                Y_coords, 
                linewidth = 0.5,
                color='lightgray',zorder=1)
        colors = {'week_peak_before':color_platte[1],'week_peak_after':color_platte[7]}

        countyvis_before = countyvis[countyvis['variable']=='week_peak_before']
        countyvis_after = countyvis[countyvis['variable']=='week_peak_after']
        plt.scatter(countyvis_before.value, 
                    countyvis_before.Power,
                    s=10,
                    c=color_platte[1],zorder=2,label='           ')

        plt.scatter(countyvis_after.value, 
                    countyvis_after.Power,
                    s=10,
                    c=color_platte[5],zorder=2,label='           ')

        plt.plot([0,10000],[0,10000],ls="--", c=".3", label='           ')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(0,10000)
        plt.ylim(0,10000)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('           ', x=0.33,y=0.95,size=fsize)
        plt.xlabel("           ", size=fsize)
        plt.ylabel("           ", size=fsize)
        plt.savefig(os.path.join(self.figure_folder_name,'fig3_scatter_peak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        countyvis = pd.melt(contyheat, id_vars=['zipcode','Power'], value_vars=['week_offpeak_before', 'week_offpeak_after'])
        X_coords = np.array([contyheat.week_offpeak_before.values,contyheat.week_offpeak_after.values])
        Y_coords = np.array([contyheat.Power.values,contyheat.Power.values])
        plt.figure(figsize=(3,6.5))
        plt.plot(X_coords, 
                Y_coords, 
                linewidth = 0.5,
                color='lightgray',zorder=1)

        countyvis_before = countyvis[countyvis['variable']=='week_offpeak_before']
        countyvis_after = countyvis[countyvis['variable']=='week_offpeak_after']

        plt.scatter(countyvis_before.value, 
                    countyvis_before.Power,
                    s=10,
                    c=color_platte[1],zorder=2,label='              ')

        plt.scatter(countyvis_after.value, 
                    countyvis_after.Power,
                    s=10,
                    c=color_platte[5],zorder=2,label='              ')

        plt.plot([0,10000],[0,10000],ls="--", c=".3", label='              ')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(0,10000)
        plt.ylim(0,10000)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('           ', fontsize=fsize,x=0.38,y=0.95)
        plt.xlabel("           ", size=fsize)
        plt.ylabel("           ", size=fsize)
        # plt.legend(frameon=False,loc='lower right',fontsize = fsize, ncol=3, bbox_to_anchor=(1., 1.0))
        plt.savefig(os.path.join(self.figure_folder_name,'fig3_scatter_offpeak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

    def visFutureCat(self):
        future_rate = 3
        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']>future_rate]

        peakridge = demand_supply_results.copy()
        peakridge['shifting'] = ' '
        peakridge['day'] = 'Grid Peak Hours'
        peakridge['peak value'] = peakridge['week_peak_before']/1000
        peakridges = peakridge.copy()

        peakridge['shifting'] = '   '
        peakridge['day'] = 'Grid Peak Hours'
        peakridge['peak value'] = peakridge['week_peak_after']/1000
        peakridges = pd.concat([peakridges,peakridge.copy()])

        offpeakridge = demand_supply_results.copy()
        offpeakridge['shifting'] = ' '
        offpeakridge['day'] = 'Grid Off-Peak Hours'
        offpeakridge['peak value'] = offpeakridge['week_offpeak_before']/1000
        offpeakridges = offpeakridge.copy()

        offpeakridge['shifting'] = '   '
        offpeakridge['day'] = 'Grid Off-Peak Hours'
        offpeakridge['peak value'] = offpeakridge['week_offpeak_after']/1000
        offpeakridges = pd.concat([offpeakridges,offpeakridge.copy()])

        g = sns.catplot(
            data=peakridges, x="adoption rate", y="peak value", hue="shifting",
            capsize=.2, palette="YlOrRd_d", errorbar=None,
            kind="point", height=3, aspect=1.3, legend=False
        )
        g.set_xlabels('               ', fontsize=fsize)
        g.set_ylabels('               ', fontsize=fsize)
        g.set_titles('               ', fontsize=fsize)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.legend(frameon=False, fontsize=fsize, loc=[0.1,0.7]).set_title('')
        plt.ylim([0,25])
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_cat_peak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        g = sns.catplot(
            data=offpeakridges, x="adoption rate", y="peak value", hue="shifting", 
            capsize=.2, palette="YlGnBu_d", errorbar=None,
            kind="point", height=3, aspect=1.3, legend=False
        )
        g.set_xlabels('               ', fontsize=fsize)
        g.set_ylabels('               ', fontsize=fsize)
        g.set_titles('               ', fontsize=fsize)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(frameon=False, fontsize=fsize, loc=[0.1,0.7]).set_title('')
        plt.ylim([0,25])
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_cat_offpeak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

    def visFutureRidge(self):
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .1, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)
            
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        demand_supply = self.supply.copy()
        ridges = demand_supply[demand_supply['adoption rate']>3]
        ridges["week_peak_before_gap"] = ridges["week_peak_before_gap"]/1000
        ridges["week_peak_after_gap"] = ridges["week_peak_after_gap"]/1000
        pal = sns.light_palette(color_platte[-2],len(ridges["adoption rate"].unique())+3,)
        g = sns.FacetGrid(ridges, row="adoption rate", hue="adoption rate", aspect=8, height=.5, palette=pal[3:])

        g.map(sns.kdeplot, "week_peak_before_gap", bw_adjust=.9, cut=5, clip_on=[-40000, 10000], alpha=0.5,  lw=2,)
        g.map(sns.kdeplot, "week_peak_after_gap", bw_adjust=.6, cut=5, clip_on=[-40000, 10000],  lw=2, linestyle="dashed",)
        g.map(plt.axhline, y=0, linewidth=5, linestyle="-", color=None, clip_on=[-40000, 10000])

        g.map(label, "adoption rate")
        g.fig.subplots_adjust(hspace=-0.7)
        g.set(yticks=[], xlabel="                      ", ylabel="", title="")
        g.despine(bottom=True, left=True)

        legend_elements = [Line2D([0], [0], color='r', lw=2, label='            ', alpha=0.5, ),
                        Line2D([0], [0], color='r', lw=2, label='            ', linestyle='dashed')]

        plt.ylabel('            ',x=-0.5,y=0.7)
        plt.legend(handles=legend_elements,frameon=False,loc='lower right',bbox_to_anchor=(1.1, 1.1))
        plt.xlim([-30,10])
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_ridge_peak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        ridges = demand_supply[demand_supply['adoption rate']>3]
        ridges["week_offpeak_before_gap"] = ridges["week_offpeak_before_gap"]/1000
        ridges["week_offpeak_after_gap"] = ridges["week_offpeak_after_gap"]/1000
        pal = sns.light_palette(color_platte[1],len(ridges["adoption rate"].unique())+3,)
        g = sns.FacetGrid(ridges, row="adoption rate", hue="adoption rate", aspect=8, height=.5, palette=pal[3:])

        g.map(sns.kdeplot, "week_offpeak_before_gap", bw_adjust=.6, cut=5, clip_on=[-40000, 10000], alpha=0.5,lw=2)
        g.map(sns.kdeplot, "week_offpeak_after_gap", bw_adjust=.6, cut=5, clip_on=[-40000, 10000], linewidth=2, linestyle="dashed",)
        g.map(plt.axhline, y=0, linewidth=5, linestyle="-", color=None, clip_on=[-40000, 10000])

        g.map(label, "adoption rate")
        g.fig.subplots_adjust(hspace=-0.7)
        g.set(yticks=[], xlabel="            ", ylabel="", title="")
        g.despine(bottom=True, left=True)

        legend_elements = [Line2D([0], [0], color='b', lw=2, label='            ', alpha=0.5, ),
                        Line2D([0], [0], color='b', lw=2, label='            ', linestyle='dashed')]

        plt.ylabel('            ',x=-0.1,y=0.7)
        plt.legend(handles=legend_elements,frameon=False,loc='lower right',bbox_to_anchor=(1.1, 1.1))
        plt.xlim([-30,10])
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_ridge_offpeak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

    def visFutureMap(self):
        comparison_rate_1 = 20; comparison_rate_2 = 100
        demand_supply = self.demand.copy()
        
        map_data = demand_supply[demand_supply['adoption rate']==comparison_rate_1]
        map_data['week_peak_diff'] = (map_data['week_peak_after']-map_data['week_peak_before'])/1000
        map_data['week_offpeak_diff'] = (map_data['week_offpeak_after']-map_data['week_offpeak_before'])/1000
        map_data['zipcode'] = map_data['zipcode'].astype(str)
        geozip_merge = self.geozip.merge(map_data,right_on='zipcode',left_on='ZCTA5CE10',how='left').copy()
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        ax= geozip_merge.plot(ax = ax, column='week_peak_diff', cmap=redcmap, legend=False, edgecolor="face", linewidth=0.1, vmin =-30)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-123, -121)
        ax.set_ylim(36.5, 38.4)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
        scatter = ax.collections[0]
        cb = plt.colorbar(scatter, ax=ax, shrink=0.6, label = '  ')
        cb.outline.set_visible(False)
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge.crs,attribution_size=6)
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_geo_adp20_peak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        map_data['zipcode'] = map_data['zipcode'].astype(str)
        geozip_merge = self.geozip.merge(map_data,right_on='zipcode',left_on='ZCTA5CE10',how='left').copy()
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        ax= geozip_merge.plot(ax = ax, column='week_offpeak_diff', cmap=homecmap, legend=False, edgecolor="face", linewidth=0.1 ,vmax = 30)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-123, -121)
        ax.set_ylim(36.5, 38.4)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
        scatter = ax.collections[0]
        cb = plt.colorbar(scatter, ax=ax, shrink=0.6, label = '  ')
        cb.outline.set_visible(False)
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge.crs,attribution_size=6)
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_geo_adp20_offpeak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        map_data = demand_supply[demand_supply['adoption rate']==comparison_rate_2]
        map_data['week_peak_diff'] = (map_data['week_peak_after']-map_data['week_peak_before'])/1000
        map_data['week_offpeak_diff'] = (map_data['week_offpeak_after']-map_data['week_offpeak_before'])/1000

        map_data['zipcode'] = map_data['zipcode'].astype(str)
        geozip_merge = self.geozip.merge(map_data,right_on='zipcode',left_on='ZCTA5CE10',how='left').copy()
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        ax= geozip_merge.plot(ax = ax, column='week_peak_diff', cmap=redcmap, legend=False, edgecolor="face", linewidth=0.1,vmin=-30)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-123, -121)
        ax.set_ylim(36.5, 38.4)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
        scatter = ax.collections[0]
        cb = plt.colorbar(scatter, ax=ax, shrink=0.6, label = '  ')
        cb.outline.set_visible(False)
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge.crs,attribution_size=6)
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_geo_adp100_peak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        map_data['zipcode'] = map_data['zipcode'].astype(str)
        geozip_merge = self.geozip.merge(map_data,right_on='zipcode',left_on='ZCTA5CE10',how='left').copy()
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        ax= geozip_merge.plot(ax = ax, column='week_offpeak_diff', cmap=homecmap, legend=False, edgecolor="face", linewidth=0.1,vmax =30)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-123, -121)
        ax.set_ylim(36.5, 38.4)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
        scatter = ax.collections[0]
        cb = plt.colorbar(scatter, ax=ax, shrink=0.6, label = '  ')
        cb.outline.set_visible(False)
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge.crs,attribution_size=6)
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_geo_adp100_offpeak.pdf'),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

    def visFutureIncrease(self):
        demand_supply = self.demand.copy()
        demand_supply = demand_supply[demand_supply['adoption rate']>3]
        peak_before = pd.pivot_table(demand_supply, values='week_peak_before', index=['zipcode'],columns=['adoption rate'] ,aggfunc=np.sum)/1000
        offpeak_before = pd.pivot_table(demand_supply, values='week_offpeak_before', index=['zipcode'],columns=['adoption rate'] ,aggfunc=np.sum)/1000
        peak_after = pd.pivot_table(demand_supply, values='week_peak_after', index=['zipcode'],columns=['adoption rate'] ,aggfunc=np.sum)/1000
        offpeak_after = pd.pivot_table(demand_supply, values='week_offpeak_after', index=['zipcode'],columns=['adoption rate'] ,aggfunc=np.sum)/1000

        up_thres = 0
        down_thres = 0

        peak_before['40-20'] = peak_before[40]-peak_before[20]; peak_before['60-40'] = peak_before[60]-peak_before[40]; 
        peak_before['80-60'] = peak_before[80]-peak_before[60]; peak_before['100-80'] = peak_before[100]-peak_before[80]
        peak_before['1'] = peak_before['60-40']-peak_before['40-20']; peak_before['2'] = peak_before['80-60']-peak_before['60-40']; peak_before['3'] = peak_before['100-80']-peak_before['80-60']
        peak_before['linear'] = 0
        peak_before.loc[(peak_before['1']<0)&(peak_before['2']<0)&(peak_before['3']<down_thres),'linear'] = -1
        peak_before.loc[(peak_before['1']>0)&(peak_before['2']>0)&(peak_before['3']>up_thres),'linear'] = 1

        offpeak_before['40-20'] = offpeak_before[40]-offpeak_before[20]; offpeak_before['60-40'] = offpeak_before[60]-offpeak_before[40]; 
        offpeak_before['80-60'] = offpeak_before[80]-offpeak_before[60]; offpeak_before['100-80'] = offpeak_before[100]-offpeak_before[80]
        offpeak_before['1'] = offpeak_before['60-40']-offpeak_before['40-20']; offpeak_before['2'] = offpeak_before['80-60']-offpeak_before['60-40']; offpeak_before['3'] = offpeak_before['100-80']-offpeak_before['80-60']
        offpeak_before['linear'] = 0
        offpeak_before.loc[(offpeak_before['1']<0)&(offpeak_before['2']<0)&(offpeak_before['3']<down_thres),'linear'] = -1
        offpeak_before.loc[(offpeak_before['1']>0)&(offpeak_before['2']>0)&(offpeak_before['3']>up_thres),'linear'] = 1

        peak_after['40-20'] = peak_after[40]-peak_after[20]; peak_after['60-40'] = peak_after[60]-peak_after[40]; 
        peak_after['80-60'] = peak_after[80]-peak_after[60]; peak_after['100-80'] = peak_after[100]-peak_after[80]
        peak_after['1'] = peak_after['60-40']-peak_after['40-20']; peak_after['2'] = peak_after['80-60']-peak_after['60-40']; peak_after['3'] = peak_after['100-80']-peak_after['80-60']
        peak_after['linear'] = 0
        peak_after.loc[(peak_after['1']<0)&(peak_after['2']<0)&(peak_after['3']<down_thres),'linear'] = -1
        peak_after.loc[(peak_after['1']>0)&(peak_after['2']>0)&(peak_after['3']>up_thres),'linear'] = 1

        offpeak_after['40-20'] = offpeak_after[40]-offpeak_after[20]; offpeak_after['60-40'] = offpeak_after[60]-offpeak_after[40]; 
        offpeak_after['80-60'] = offpeak_after[80]-offpeak_after[60]; offpeak_after['100-80'] = offpeak_after[100]-offpeak_after[80]
        offpeak_after['1'] = offpeak_after['60-40']-offpeak_after['40-20']; offpeak_after['2'] = offpeak_after['80-60']-offpeak_after['60-40']; offpeak_after['3'] = offpeak_after['100-80']-offpeak_after['80-60']
        offpeak_after['linear'] = 0
        offpeak_after.loc[(offpeak_after['1']<0)&(offpeak_after['2']<0)&(offpeak_after['3']<down_thres),'linear'] = -1
        offpeak_after.loc[(offpeak_after['1']>0)&(offpeak_after['2']>0)&(offpeak_after['3']>up_thres),'linear'] = 1

        linear_map = pd.DataFrame()
        linear_map['peak_before'] = peak_before['linear']
        linear_map['peak_after'] = peak_after['linear']
        linear_map['offpeak_before'] = offpeak_before['linear']
        linear_map['offpeak_after'] = offpeak_after['linear']
        linear_map = linear_map.reset_index()

        fig, ax = plt.subplots(figsize=(3.5,3.5))
        (peak_before[peak_before['linear']==-1][[20,40,60,80,100]]).median().plot(label='Sublinear',c = '#e0884e')
        (peak_before[peak_before['linear']==1][[20,40,60,80,100]]).median().plot(label='Superlinear', c= '#93003a')
        y1 = (peak_before[peak_before['linear']==-1][[20]]).median()
        y2 =  (peak_before[peak_before['linear']==-1][[100]]).median()
        plt.plot([20,100],[y1,y2],c='k',linestyle='dashed')
        y1 = (peak_before[peak_before['linear']==1][[20]]).median()
        y2 =  (peak_before[peak_before['linear']==1][[100]]).median()
        plt.ylim([0,30])
        plt.plot([20,100],[y1,y2],c='k',linestyle='dashed',label="Linear")
        plt.title("Grid Peak Hours: Before Shifting")
        plt.xlabel('Adoption Rate [%]')
        plt.ylabel('Charging Load [MW]')
        plt.legend(facecolor='white')
        plt.savefig(os.path.join(self.figure_folder_name,'sp_subsuper_peak_before.pdf'),format='pdf',dpi=300,bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(figsize=(3.5,3.5))
        (peak_after[peak_after['linear']==-1][[20,40,60,80,100]]).median().plot(label='Sublinear',c = '#e0884e')
        (peak_after[peak_after['linear']==1][[20,40,60,80,100]]).median().plot(label='Superlinear',c= '#93003a' ) 
        y1 = (peak_after[peak_after['linear']==-1][[20]]).median()
        y2 =  (peak_after[peak_after['linear']==-1][[100]]).median()
        plt.plot([20,100],[y1,y2],c='k',linestyle='dashed')
        y1 = (peak_after[peak_after['linear']==1][[20]]).median()
        y2 =  (peak_after[peak_after['linear']==1][[100]]).median()
        plt.ylim([0,30])
        plt.plot([20,100],[y1,y2],c='k',linestyle='dashed',label="Linear")
        plt.title("Grid Peak Hours: After Shifting")
        plt.xlabel('Adoption Rate [%]')
        plt.ylabel('Charging Load [MW]')
        plt.legend(facecolor='white')
        plt.savefig(os.path.join(self.figure_folder_name,'sp_subsuper_peak_after.pdf'),format='pdf',dpi=300,bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(figsize=(3.5,3.5))
        (offpeak_before[offpeak_before['linear']==-1][[20,40,60,80,100]]).median().plot(label='Sublinear', c =  '#c7ece7')
        (offpeak_before[offpeak_before['linear']==1][[20,40,60,80,100]]).median().plot(label='Superlinear', c = '#00429d')
        y1 = (offpeak_before[offpeak_before['linear']==-1][[20]]).median()
        y2 =  (offpeak_before[offpeak_before['linear']==-1][[100]]).median()
        plt.plot([20,100],[y1,y2],c='k',linestyle='dashed')
        y1 = (offpeak_before[offpeak_before['linear']==1][[20]]).median()
        y2 =  (offpeak_before[offpeak_before['linear']==1][[100]]).median()
        plt.ylim([0,30])
        plt.plot([20,100],[y1,y2],c='k',linestyle='dashed',label="Linear")
        plt.title("Grid Off-Peak Hours: Before Shifting")
        plt.xlabel('Adoption Rate [%]')
        plt.ylabel('Charging Load [MW]')
        plt.legend(facecolor='white')
        plt.savefig(os.path.join(self.figure_folder_name,'sp_subsuper_offpeak_before.pdf'),format='pdf',dpi=300,bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(figsize=(3.5,3.5))
        (offpeak_after[offpeak_after['linear']==-1][[20,40,60,80,100]]).median().plot(label='Sublinear', c = '#c7ece7')
        (offpeak_after[offpeak_after['linear']==1][[20,40,60,80,100]]).median().plot(label='Superlinear', c = '#00429d')
        y1 = (offpeak_after[offpeak_after['linear']==-1][[20]]).median()
        y2 =  (offpeak_after[offpeak_after['linear']==-1][[100]]).median()
        plt.plot([20,100],[y1,y2],c='k',linestyle='dashed')
        y1 = (offpeak_after[offpeak_after['linear']==1][[20]]).median()
        y2 =  (offpeak_after[offpeak_after['linear']==1][[100]]).median()
        plt.ylim([0,30])
        plt.plot([20,100],[y1,y2],c='k',linestyle='dashed',label="Linear")
        plt.title("Grid Off-Peak Hours: After Shifting")
        plt.xlabel('Adoption Rate [%]')
        plt.ylabel('Charging Load [MW]')
        plt.legend(facecolor='white')
        plt.savefig(os.path.join(self.figure_folder_name,'sp_subsuper_offpeak_after.pdf'),format='pdf',dpi=300,bbox_inches='tight')
        plt.show()

        linear_map['zipcode'] = linear_map['zipcode'].astype(int).astype(str)
        geozip_merge = self.geozip.merge(linear_map,right_on='zipcode',left_on='ZCTA5CE10',how='left').copy()

        fig, ax = plt.subplots(figsize=(6,6),nrows=1, ncols=2)

        geozip_merge_sup = geozip_merge[geozip_merge['peak_before']>0]
        ax[0] = geozip_merge_sup.plot(ax = ax[0], legend=False, edgecolor="face", linewidth=0.1, color = 'r')
        geozip_merge_sub = geozip_merge[geozip_merge['peak_before']<0]
        ax[0] = geozip_merge_sub.plot(ax = ax[0], legend=False, edgecolor="face", linewidth=0.1, color = 'b')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_xlim(-123, -121)
        ax[0].set_ylim(36.5, 38.4)
        cx.add_basemap(ax[0], source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge.crs,attribution_size=6)

        geozip_merge_sup = geozip_merge[geozip_merge['offpeak_before']>0]
        ax[1] = geozip_merge_sup.plot(ax = ax[1], legend=False, edgecolor="face", linewidth=0.1, color = 'r')
        geozip_merge_sub = geozip_merge[geozip_merge['offpeak_before']<0]
        ax[1] = geozip_merge_sub.plot(ax = ax[1], legend=False, edgecolor="face", linewidth=0.1, color = 'b')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_xlim(-123, -121)
        ax[1].set_ylim(36.5, 38.4)
        cx.add_basemap(ax[1], source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge.crs,attribution_size=6)
        plt.savefig(os.path.join(self.figure_folder_name,'sp_geo_linear_peak.pdf'),dpi=300,transparent=True, bbox_inches='tight')
        plt.show()

        linear_map['zipcode'] = linear_map['zipcode'].astype(int).astype(str)
        geozip = gpd.read_file(os.path.join('..','data','census','sfbay_zip.geojson')) 
        geozip_merge = geozip.merge(linear_map,right_on='zipcode',left_on='ZCTA5CE10',how='left') 
        fig, ax = plt.subplots(figsize=(6,6),nrows=1, ncols=2)

        geozip_merge_sup = geozip_merge[geozip_merge['peak_after']>0]
        ax[0] = geozip_merge_sup.plot(ax = ax[0], legend=False, edgecolor="face", linewidth=0.1, color = 'r')
        geozip_merge_sub = geozip_merge[geozip_merge['peak_after']<0]
        ax[0] = geozip_merge_sub.plot(ax = ax[0], legend=False, edgecolor="face", linewidth=0.1, color = 'b')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_xlim(-123, -121)
        ax[0].set_ylim(36.5, 38.4)
        cx.add_basemap(ax[0], source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge.crs,attribution_size=6)

        geozip_merge_sup = geozip_merge[geozip_merge['offpeak_after']>0]
        ax[1] = geozip_merge_sup.plot(ax = ax[1], legend=False, edgecolor="face", linewidth=0.1, color = 'r')
        geozip_merge_sub = geozip_merge[geozip_merge['offpeak_after']<0]
        ax[1] = geozip_merge_sub.plot(ax = ax[1], legend=False, edgecolor="face", linewidth=0.1, color = 'b')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_xlim(-123, -121)
        ax[1].set_ylim(36.5, 38.4)
        cx.add_basemap(ax[1], source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge.crs,attribution_size=6)
        plt.savefig(os.path.join(self.figure_folder_name,'sp_geo_linear_offpeak.pdf'),dpi=300,transparent=True, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    anashf = AnaShift(0.5,0.54,17,21,1,9)
    anashf.visTempLoad()
    anashf.visShaveSector()
    anashf.visStation()
    anashf.visZIPPair()
    anashf.visZipScatter()
    anashf.visFutureRidge()
    anashf.visFutureMap()
    anashf.visFutureIncrease()
    anashf.visFutureCat()

