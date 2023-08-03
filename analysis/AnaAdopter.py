import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import contextily as cx
from sklearn import linear_model
from scipy.stats import pearsonr
import matplotlib.colors as clr
import warnings
warnings.filterwarnings("ignore")

mode = 1; year= 0
fsize = 12; tdir = 'in'; major = 6; minor = 3; lwidth = 1; lhandle = 1
space_symbol = '                          '
mymarkersize  = 4
color_platte = ['#00429d', '#4771b2', 
'#73a2c6', '#a5d5d8', '#ffffe0', '#fcc17e', '#e0884e', '#bc4f3c', '#93003a']
plt.style.use('default')
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

colors= ['#00429d', '#73a2c6', '#ffffe0', '#e0884e', '#93003a']
nodes = np.arange(0,1.1,0.25)
catcmap = clr.LinearSegmentedColormap.from_list("catcmap", list(zip(nodes, colors)))

class AnaAdopter():
    def __init__(self, work_l2_access,home_l2_access,peak_start,peak_end,acceptance,max_stay):
        super().__init__()
        self.work_l2_access = work_l2_access
        self.home_l2_access = home_l2_access
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.acceptance = acceptance
        self.max_stay = max_stay
        self.end = pd.to_datetime('2019-12-31')
        self.mobility_folder_name = os.path.join('..','result','mobility')
        self.adopter_folder_name = os.path.join('..','result','adopter','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access))
        self.behavior_folder_name = os.path.join('..','result','behavior','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access))
        self.shift_folder_name = os.path.join('..','result','shift','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access)+'peak_'+str(self.peak_start)+'_'+str(self.peak_end)+'acceptance_'+str(self.acceptance)+'max_stay_'+str(self.max_stay))
        self.figure_folder_name = os.path.join('..','figure','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access)+'peak_'+str(self.peak_start)+'_'+str(self.peak_end)+'acceptance_'+str(self.acceptance)+'max_stay_'+str(self.max_stay))
        self.cvrp = pd.read_excel(os.path.join('..','data','census','CVRPStats.xlsx'))
        self.geozip = gpd.read_file(os.path.join('..','data','census','sfbay_zip.geojson'))
        self.df_user_inf = pickle.load(open(os.path.join(self.adopter_folder_name,'df_user_inf.pkl'), 'rb'), encoding='bytes')
        if not os.path.exists(self.figure_folder_name):
            os.makedirs(self.figure_folder_name)
    
    def visAccess(self):
        self.df_user_inf['personAccess'].hist()
        plt.show()

    def visMarketShare(self):
        self.cvrp['PURCHASE_DATE'] = pd.to_datetime(self.cvrp['PURCHASE_DATE'])

        cvrp_cat = self.cvrp[(self.cvrp['AIR_DIST']=='Bay Area')&(self.cvrp['PURCHASE_DATE']<self.end)][['ID','VEH_CAT']]
        cvrp_cat = pd.pivot_table(cvrp_cat, values='ID', columns ='VEH_CAT',aggfunc=np.count_nonzero)
        print('BEV ratio',cvrp_cat['BEV'].values/np.sum(cvrp_cat.values))

        cvpr_bev = self.cvrp[(self.cvrp['AIR_DIST']=='Bay Area')&(self.cvrp['PURCHASE_DATE']<self.end)&(self.cvrp['VEH_CAT']=='BEV')]
        cvpr_oem = pd.pivot_table(cvpr_bev, values='ID', index = 'VEH_DET', aggfunc=np.count_nonzero)
        cvrp_vis = cvpr_oem.sort_values(by='ID',ascending=False)
        fig, ax = plt.subplots(figsize=(12,2))
        plt.bar(cvrp_vis.index,cvrp_vis.ID)
        plt.xticks(rotation = 90)
        ax.set_ylabel('# Model')
        ax.set_ylabel('Model Type')
        plt.savefig(os.path.join(self.figure_folder_name,'sp_cvrp_model.pdf'), dpi = 300, pad_inches = .1, bbox_inches = 'tight')
        plt.show()

    def visValidation(self):
        cvpr_bay = self.cvrp[(self.cvrp['AIR_DIST']=='Bay Area')&(self.cvrp['PURCHASE_DATE']<self.end)]
        cvpr_bay_table = pd.pivot_table(cvpr_bay, values=['ID'], index=['ZIP'], aggfunc=np.count_nonzero)/0.7
        cvpr_bay_table['ID'] = cvpr_bay_table['ID'].astype(int)
        cvpr_bay_table = cvpr_bay_table.reset_index()
        cvpr_bay_table['ZIP'] = cvpr_bay_table['ZIP'].astype(str)
        geozip_merge_cvrp = self.geozip.merge(cvpr_bay_table,right_on='ZIP',left_on='ZCTA5CE10',how='left')

        user_list = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(1) + 'p.pkl'), 'rb'), encoding='bytes')
        df_user_inf_select = self.df_user_inf[self.df_user_inf['userID'].isin(user_list)]
        df_user = pd.pivot_table(df_user_inf_select[['userID','homeZipcode']], values='userID', index=['homeZipcode'],aggfunc=np.count_nonzero).reset_index()
        
        df_user['homeZipcode'] = df_user['homeZipcode'].astype(str)
        geozip_merge_simulated = self.geozip.merge(df_user,right_on='homeZipcode',left_on='ZCTA5CE10',how='left') 
        common_list = set(geozip_merge_cvrp['ZIP'])&set(geozip_merge_simulated['homeZipcode'].dropna())
        geozip_merge_simulated = geozip_merge_simulated[geozip_merge_simulated['homeZipcode'].isin(common_list)]
        geozip_merge_cvrp = geozip_merge_cvrp[geozip_merge_cvrp['ZIP'].isin(common_list)]

        fig, ax = plt.subplots(figsize=(4,4))
        ax = geozip_merge_cvrp.plot(ax = ax, column='ID', cmap='YlGnBu',edgecolor="face", linewidth=0.1, vmin=0, vmax = 6000)
        cx.add_basemap(ax,source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge_cvrp.crs)
        ax.set_xticks([]); ax.set_yticks([])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
        scatter = ax.collections[0]
        cb = plt.colorbar(scatter, ax=ax, shrink=0.6, label = "# EV (CVRP)")
        cb.outline.set_visible(False)
        ax.grid()
        plt.savefig(os.path.join(self.figure_folder_name,'sp_cvrp_maptruth.pdf'), dpi = 300, pad_inches = .1, bbox_inches = 'tight')
        plt.show()

        fig, axs = plt.subplots(figsize=(4,4))
        ax = geozip_merge_simulated.plot(ax = axs, column='userID', cmap='YlOrRd',edgecolor="face", linewidth=0.1,vmin=0, vmax = 6000)
        ax.set_xticks([])
        ax.set_yticks([])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
        scatter = ax.collections[0]
        cb = plt.colorbar(scatter, ax=ax, shrink=0.6, label = "# EV (Simulated)")
        cb.outline.set_visible(False)
        ax.grid()
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge_simulated.crs)
        plt.savefig(os.path.join(self.figure_folder_name,'sp_cvrp_mapsim.pdf'), dpi = 300, pad_inches = .1, bbox_inches = 'tight')
        plt.show()

        cor_cv_sim = pd.DataFrame()
        cor_cv_sim['sim'] = geozip_merge_simulated['userID'].fillna(0)
        cor_cv_sim['cvrp'] = geozip_merge_cvrp['ID'].fillna(0)
        cor_plot = cor_cv_sim[(cor_cv_sim['sim']!=0)|(cor_cv_sim['cvrp']!=0)]

        regr = linear_model.LinearRegression(fit_intercept=True)
        X = cor_plot['cvrp'].values.reshape(-1,1)
        y = cor_plot['sim'].values.reshape(-1,1)
        regr.fit(X, y)
        print('reg_coef: ',regr.coef_[0])
        print('reg_intercept: ',regr.intercept_)
        # print('reg_score: ',regr.score(X, y))
        # print(pearsonr(cor_plot['cvrp'].values, cor_plot['sim'].values))
        g = sns.jointplot(x="cvrp", y="sim", data=cor_plot,
                        kind="reg", truncate=True,
                        xlim=(0, 5000), ylim=(0, 5000),height=4)
        g.ax_joint.set_xlabel('# EV (CVRP)')
        g.ax_joint.set_ylabel('# EV (Simulated)')
        left, width = .7, .3
        bottom, height = .3, .3
        right = left + width
        top = bottom + height
        plt.text(left, top, '$y=1.06x+2.72$',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        plt.text(left, top-0.1, 'Correlation $=0.84$',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        plt.savefig(os.path.join(self.figure_folder_name,'sp_cvrp_hist.pdf'), dpi = 300,format='pdf',  bbox_inches = 'tight')
        plt.show()
    
    def visDemographics(self):
        df_total_list = []
        for (year,rate) in zip(range(2,7),[20,40,60,80,100]):
            user_list = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(year) + 'p.pkl'), 'rb'), encoding='bytes')
            df_user_inf_select = self.df_user_inf[self.df_user_inf['userID'].isin(user_list)]
            df_user_inf_select['Adoption Rate [%]'] = rate
            df_user_inf_select['Daily Travel Distance [Miles]'] = df_user_inf_select['personDistance']
            df_user_inf_select['Household Income [1,000 $]'] = df_user_inf_select['personIncome']/1000
            df_total_list.append(df_user_inf_select)
        df_total = pd.concat(df_total_list)

        sns.set_theme(style="white", palette=None)
        fig, ax = plt.subplots(figsize=(4,2.5))
        g = sns.boxplot(x='Household Income [1,000 $]',y='Adoption Rate [%]',data=df_total, notch=True,orient='h',showfliers=False,linewidth=1)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        plt.xlabel(space_symbol)
        plt.ylabel(space_symbol)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_box_income.pdf'),format='pdf',dpi=300,bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(figsize=(4,2.5))
        g = sns.boxplot(x='Daily Travel Distance [Miles]',y='Adoption Rate [%]',data=df_total, notch=True,orient='h',showfliers=False, linewidth=1)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel(space_symbol)
        plt.ylabel(space_symbol)
        plt.savefig(os.path.join(self.figure_folder_name,'fig4_box_distance.pdf'),format='pdf',dpi=300,bbox_inches='tight')
        plt.show()

    def visAdoption(self):

        select_user_info = {}
        for year in [1,2,3,4,5,6]:
            user_list = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(year) + 'p.pkl'), 'rb'), encoding='bytes')
            df_user_inf_select = self.df_user_inf[self.df_user_inf['userID'].isin(user_list)]
            select_user_info[year] = df_user_inf_select

            df_user = pd.pivot_table(df_user_inf_select, values='userID', index=['homeZipcode'],aggfunc=np.count_nonzero).reset_index()
            geozip = gpd.read_file(os.path.join('..','data','census','sfbay_zip.geojson'))

            df_user['homeZipcode'] = df_user['homeZipcode'].astype(str)
            geozip_merge_simulated = geozip.merge(df_user,right_on='homeZipcode',left_on='ZCTA5CE10',how='left') 

            fig, axs = plt.subplots(figsize=(4,4))
            ax = geozip_merge_simulated.plot(ax = axs, column='userID', cmap='YlOrRd',edgecolor="face", linewidth=0.1, vmin=0, vmax=60000)
            ax.set_xticks([])
            ax.set_yticks([])
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1)
            scatter = ax.collections[0]
            cb = plt.colorbar(scatter, ax=ax, shrink=0.6, label = "# EV (Simulated)")
            cb.outline.set_visible(False)
            ax.grid()
            cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, crs=geozip_merge_simulated.crs)
            plt.savefig(os.path.join(self.figure_folder_name,'sp_cvrp_mapsim_'+str(year)+'.pdf'), dpi = 300, pad_inches = .1, bbox_inches = 'tight')
            plt.show()

if __name__ == "__main__":
    anadp = AnaAdopter(0.5,0.54,17,21,1,9)
    anadp.visValidation()
    anadp.visAccess()
    anadp.visMarketShare()
    anadp.visDemographics()
    anadp.visAdoption()
    



    
