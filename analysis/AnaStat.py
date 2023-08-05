import pickle
import numpy as np
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

class AnaStat():
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
        self.df_user_inf = pickle.load(open(os.path.join(self.adopter_folder_name,'df_user_inf.pkl'), 'rb'), encoding='bytes')
        self.df_user_group = pd.read_csv(os.path.join(self.behavior_folder_name,'df_user_group.csv'))
        self.df_user_session = pd.read_csv(os.path.join(self.behavior_folder_name,'simulated_session.csv'))
        self.df_shift_session = pd.read_csv(os.path.join(self.shift_folder_name,'shifted_session.csv')) 
        self.demand = pd.read_csv(os.path.join(self.shift_folder_name,'demand.csv'))
        self.supply = pd.read_csv(os.path.join(self.shift_folder_name,'supply.csv'))
        self.getPeakHour()
        if not os.path.exists(self.figure_folder_name):
            os.makedirs(self.figure_folder_name)

    def getPeakHour(self):
        peak_start = self.peak_start; peak_end=self.peak_end
        week_peak = []; week_offpeak = []; weekend = []; 
        for hour in range(168):
            if hour<24*5:
                if (hour%24 >= peak_start) and (hour%24 < peak_end):
                    week_peak.append(hour)
                else:
                    week_offpeak.append(hour)
            else:
                weekend.append(hour)
        self.week_peak = week_peak
        self.week_offpeak = week_offpeak
        self.weekend = weekend

    def simEVLoad(self,df):
        zipcode_list = self.df_user_session['stay_zipcode'].dropna().unique().copy()

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
    
    def calChargingGroup(self): #2a
        print("=====================Figure 2A=======================")
        p_z = {}
        for nG in range(136):
            p_z[nG] = pd.read_csv(os.path.join('..','data','speech','SPEECh Original Model 136','pz_weekday_g_'+str(nG)+'.csv'),index_col=0).mean()
        group_feature = pd.DataFrame.from_dict(p_z).T
        group_feature['home'] = group_feature['home_l2 - Fraction of weekdays with session'] + group_feature['home_l1 - Fraction of weekdays with session'] + group_feature['mud_l2 - Fraction of weekdays with session']
        group_feature['work'] = group_feature['work_l2 - Fraction of weekdays with session'] 
        group_feature['other'] = group_feature['public_l2 - Fraction of weekdays with session'] + group_feature['public_l3 - Fraction of weekenddays with session']
        group_feature['total'] = group_feature['home']+group_feature['work']+group_feature['other']+1e-5
        group_feature['home_norm'] = group_feature['home']/group_feature['total']
        group_feature['work_norm'] = group_feature['work']/group_feature['total']
        group_feature['other_norm'] = group_feature['other']/group_feature['total']
        group_feature['home_label'] = 0; group_feature['work_label'] = 0; group_feature['other_label'] = 0; group_feature['mix_label'] = 0
        threshold = 0.3
        group_feature.loc[group_feature['home_norm']>threshold,'home_label'] = 1
        group_feature.loc[group_feature['work_norm']>threshold,'work_label'] = 1
        group_feature.loc[group_feature['other_norm']>threshold,'other_label'] = 1
        group_feature['total_label'] = group_feature['other_label']+group_feature['home_label']+group_feature['work_label']
        group_feature.loc[group_feature['total_label']>=2,'mix_label'] = 1
        group_feature['new_label'] = group_feature[['home_norm','work_norm','other_norm']].idxmax(axis=1)
        group_feature.loc[group_feature['mix_label']==1,'new_label'] = 'mix_norm'
        group = self.df_user_group.merge(group_feature['new_label'],left_on='personGroup',right_on=group_feature['new_label'].index)
        group['new_label'] = group['new_label'].map({'home_norm': 'Home', 'work_norm': 'Work', 'other_norm': 'Public', 'mix_norm':'Mix'})

        group_bar = pd.pivot_table(group, values='userID', columns ='new_label',aggfunc=np.count_nonzero).sum()/1000000
        group_bar = group_bar.reset_index()
        group_number = group_bar[0]/sum(group_bar[0])
        print('Home Charging Group Type: ',group_number.values[0])
        print('Work Charging Group Type: ',group_number.values[3])
        print('Public Charging Group Type: ',group_number.values[2])
        print('Mix Charging Group Type: ',group_number.values[1])

    def calDemandTemporal(self):
        print("=====================Figure 2C=======================")
        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==2]
        print('ZIP Code level average on-peak load [MW]',np.mean(demand_supply_results['week_peak_before'])/1000)
        print('ZIP Code level average off-peak load [MW]',np.mean(demand_supply_results['week_offpeak_before'])/1000)

    def calDemandSpatial(self):
        print("=====================Figure 2D=======================")
        user_select = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(0) + 'p.pkl'), 'rb'), encoding='bytes')
        df = self.df_user_session[self.df_user_session['id'].isin(user_select)].copy()
        home, work, other, total = self.simEVLoad(df)
        total['hour_sum_o'] = total.sum(axis=1)
        print('ZIP Code level weekly total charging needs [MWh], mean:',np.mean(total['hour_sum_o']/1000),', std: ',np.std(total['hour_sum_o']/1000))
        home['hour_sum_o'] = home.sum(axis=1)
        print('ZIP Code level max weekly home charging needs [MWh], mean:',np.max(home['hour_sum_o']/1000,axis=0))
        work['hour_sum_o'] = work.sum(axis=1)
        print('ZIP Code level max weekly work charging needs [MWh], mean:',np.max(work['hour_sum_o']/1000,axis=0))
        other['hour_sum_o'] = other.sum(axis=1)
        print('ZIP Code level max weekly other charging needs [MWh], mean:',np.max(other['hour_sum_o']/1000,axis=0))

    def calShaveTemporal(self):
        print("=====================Figure 3A=======================")
        df_user_session = self.df_user_session.copy()
        df_shift_session = self.df_shift_session.copy()
        user_select = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(0) + 'p.pkl'), 'rb'), encoding='bytes')
        df_before = df_user_session[df_user_session['id'].isin(user_select)]
        df_after = df_shift_session[df_shift_session['id'].isin(user_select)]

        df_before = df_before.fillna(-1)
        df_after = df_after.fillna(-1)
        df_before = pd.pivot_table(df_before,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()
        df_after = pd.pivot_table(df_after,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()
        home_load, work_load, other_load, total_load = self.simEVLoad(df_before)
        homes_load, works_load, others_load, totals_load = self.simEVLoad(df_after)
        
        week_peak_total = total_load.values[:,self.week_peak]
        week_offpeak_total = total_load.values[:,self.week_offpeak]

        week_peak_totals = totals_load.values[:,self.week_peak]
        week_offpeak_totals = totals_load.values[:,self.week_offpeak]

        print("Total charging peak load reduction of total",(np.max(np.sum(week_peak_total,axis=0))-np.max(np.sum(week_peak_totals,axis=0)))/1000)
        print("Total charging off-peak load reduction of total",(np.max(np.sum(week_offpeak_total,axis=0))-np.max(np.sum(week_offpeak_totals,axis=0)))/1000)

        peak_hour_before, peak_hour_after= np.argmax(np.sum(week_peak_total,axis=0)),np.argmax(np.sum(week_peak_totals,axis=0))
        offpeak_hour_before, offpeak_hour_after= np.argmax(np.sum(week_offpeak_total,axis=0)),np.argmax(np.sum(week_offpeak_totals,axis=0))

        week_peak_home = home_load.values[:,self.week_peak]
        week_offpeak_home = home_load.values[:,self.week_offpeak]

        week_peak_homes = homes_load.values[:,self.week_peak]
        week_offpeak_homes = homes_load.values[:,self.week_offpeak]

        print("Home charging peak load reduction of total",(np.sum(week_peak_home,axis=0)[peak_hour_before]-np.sum(week_peak_homes,axis=0)[peak_hour_after])/(np.max(np.sum(week_peak_total,axis=0))-np.max(np.sum(week_peak_totals,axis=0))))
        print("Home charging off-peak load reduction of total",-(np.sum(week_offpeak_home,axis=0)[offpeak_hour_before]-np.sum(week_offpeak_homes,axis=0)[offpeak_hour_after])/(np.max(np.sum(week_offpeak_total,axis=0))-np.max(np.sum(week_offpeak_totals,axis=0))))

        week_peak_work = work_load.values[:,self.week_peak]
        week_offpeak_work = work_load.values[:,self.week_offpeak]

        week_peak_works = works_load.values[:,self.week_peak]
        week_offpeak_works = works_load.values[:,self.week_offpeak]

        print("Work charging peak load reduction of total",(np.sum(week_peak_work,axis=0)[peak_hour_before]-np.sum(week_peak_works,axis=0)[peak_hour_after])/(np.max(np.sum(week_peak_total,axis=0))-np.max(np.sum(week_peak_totals,axis=0))))
        print("Work charging off-peak load reduction of total",-(np.sum(week_offpeak_work,axis=0)[offpeak_hour_before]-np.sum(week_offpeak_works,axis=0)[offpeak_hour_after])/(np.max(np.sum(week_offpeak_total,axis=0))-np.max(np.sum(week_offpeak_totals,axis=0))))

        week_peak_other = other_load.values[:,self.week_peak]
        week_offpeak_other = other_load.values[:,self.week_offpeak]

        week_peak_others = others_load.values[:,self.week_peak]
        week_offpeak_others = others_load.values[:,self.week_offpeak]

        print("Other charging peak load reduction of total",(np.sum(week_peak_other,axis=0)[peak_hour_before]-np.sum(week_peak_others,axis=0)[peak_hour_after])/(np.max(np.sum(week_peak_total,axis=0))-np.max(np.sum(week_peak_totals,axis=0))))
        print("Other charging off-peak load reduction of total",-(np.sum(week_offpeak_other,axis=0)[offpeak_hour_before]-np.sum(week_offpeak_others,axis=0)[offpeak_hour_after])/(np.max(np.sum(week_offpeak_total,axis=0))-np.max(np.sum(week_offpeak_totals,axis=0))))
    
    def calShaveSector(self): 
        print("=====================Figure 3B=======================")
        df_shift_session = self.df_shift_session.copy()
        user_select = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(0) + 'p.pkl'), 'rb'), encoding='bytes')
        df_shift_session = df_shift_session[df_shift_session['id'].isin(user_select)]
        df_shift_total = df_shift_session[df_shift_session['is_shift']!=0]

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

        print('Home - Home:', home_home/home_sum, 'Home - Work:', home_work/home_sum, 'Home - Public:',home_public/home_sum)
        print('Work - Home:', work_home/work_sum, 'Work - Work:', work_work/work_sum, 'Work - Public:',work_public/work_sum)
        print('Public - Home:', public_home/public_sum, 'Public - Work:', public_work/public_sum, 'Public - Public:',public_public/public_sum)
        
    def calSupplyShave(self):
        print("=====================Figure 3C=======================")
        demand_supply = self.supply.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==2]
        print('ZIP Code level average reduction in on-peak hours [MW]',np.mean(demand_supply_results['week_peak_after_gap']-demand_supply_results['week_peak_before_gap'])/1000)
        print('ZIP Code level average reduction in off-peak hours [MW]',np.mean(demand_supply_results['week_offpeak_after_gap']-demand_supply_results['week_offpeak_before_gap'])/1000)

        print("=====================Figure 4B=======================")
        print('# ZIP Code with insufficient charging stations during offpeak hours before recommendations',len(demand_supply_results[demand_supply_results['week_offpeak_before_gap']<0]))
        print('# ZIP Code with insufficient charging stations during peak hours before recommendations',len(demand_supply_results[demand_supply_results['week_peak_before_gap']<0]))
        print('# ZIP Code with insufficient charging stations during offpeak hours after recommendations',len(demand_supply_results[demand_supply_results['week_offpeak_after_gap']<0]))
        print('# ZIP Code with insufficient charging stations during peak hours after recommendations',len(demand_supply_results[demand_supply_results['week_peak_after_gap']<0]))
        print('Current Charging capacity - Needed charging capacity [MW]', np.sum(demand_supply_results['week_peak_before_gap'])/1000)

    def calFutureDemo(self):
        df_total_list = []
        for (year,rate) in zip(range(2,7),[20,40,60,80,100]):
            user_list = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(year) + 'p.pkl'), 'rb'), encoding='bytes')
            df_user_inf_select = self.df_user_inf[self.df_user_inf['userID'].isin(user_list)]
            df_user_inf_select['Adoption Rate [%]'] = rate
            df_user_inf_select['Daily Travel Distance [Miles]'] = df_user_inf_select['personDistance']
            df_user_inf_select['Household Income [1,000 $]'] = df_user_inf_select['personIncome']/1000
            df_total_list.append(df_user_inf_select)
        df_total = pd.concat(df_total_list)
        print("=====================Figure 5A=======================")
        print('Household income [1,000 $], adoption rate 20%, median: ',df_total[df_total['Adoption Rate [%]']==20]['Household Income [1,000 $]'].median(),'std: ',df_total[df_total['Adoption Rate [%]']==20]['Household Income [1,000 $]'].std())
        print('Household income [1,000 $], adoption rate 100%, median: ',df_total[df_total['Adoption Rate [%]']==100]['Household Income [1,000 $]'].median(),'std: ',df_total[df_total['Adoption Rate [%]']==100]['Household Income [1,000 $]'].std())
        print("=====================Figure 5B=======================")
        print('Daily Travel Distance [Miles], adoption rate 20%, median: ',df_total[df_total['Adoption Rate [%]']==20]['Daily Travel Distance [Miles]'].median(),'std: ',df_total[df_total['Adoption Rate [%]']==20]['Daily Travel Distance [Miles]'].std())
        print('Daily Travel Distance [Miles], adoption rate 100%, median: ',df_total[df_total['Adoption Rate [%]']==100]['Daily Travel Distance [Miles]'].median(),'std: ',df_total[df_total['Adoption Rate [%]']==100]['Daily Travel Distance [Miles]'].std())
    
    def calFutureDemandSpatial(self):
        print("=====================Figure 5C=======================")
        thres = 0.3
        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==20]
        demand_supply_results['reduce_ratio'] = (demand_supply_results['week_peak_before']-demand_supply_results['week_peak_after'])/demand_supply_results['week_peak_before']
        print('# ZIP Code reducing peak hour charging loads by at least 30%, adoption rate 20%,',len(demand_supply_results[demand_supply_results['reduce_ratio']>thres]))

        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==100]
        demand_supply_results['reduce_ratio'] = (demand_supply_results['week_peak_before']-demand_supply_results['week_peak_after'])/demand_supply_results['week_peak_before']
        print('# ZIP Code reducing peak hour charging loads by at least 30%, adoption rate 100%,',len(demand_supply_results[demand_supply_results['reduce_ratio']>thres]))

        thres = 0.3
        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==20]
        demand_supply_results['reduce_ratio'] = (demand_supply_results['week_offpeak_after']-demand_supply_results['week_offpeak_before'])/demand_supply_results['week_offpeak_before']
        print('# ZIP Code reducing offpeak hour charging loads by at least 30%, adoption rate 20%,',len(demand_supply_results[demand_supply_results['reduce_ratio']>thres]))

        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==100]
        demand_supply_results['reduce_ratio'] = (demand_supply_results['week_offpeak_after']-demand_supply_results['week_offpeak_before'])/demand_supply_results['week_offpeak_before']
        print('# ZIP Code reducing offpeak hour charging loads by at least 30%, adoption rate 100%,', len(demand_supply_results[demand_supply_results['reduce_ratio']>thres]))

    def calFutureDemandTemporal(self):
        print("=====================Figure 5D=======================")
        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==20]
        print('ZIP code level average reduction ratio in peak loads, adoption rate 20%,',np.mean((demand_supply_results['week_peak_after']-demand_supply_results['week_peak_before'])/demand_supply_results['week_peak_before']))
        print('ZIP code level average reduction in peak loads, adoption rate 20%,',np.mean(demand_supply_results['week_peak_after']-demand_supply_results['week_peak_before']))

        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==100]
        print('ZIP code level average reduction ratio in peak loads, adoption rate 100%,',np.mean((demand_supply_results['week_peak_after']-demand_supply_results['week_peak_before'])/demand_supply_results['week_peak_before']))
        print('ZIP code level average reduction in peak loads, adoption rate 100%,',np.mean(demand_supply_results['week_peak_after']-demand_supply_results['week_peak_before']))

        print("=====================Figure 5E=======================")
        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==20]
        demand_supply_results_ratio = demand_supply_results[demand_supply_results['week_offpeak_before']>0]
        print('ZIP code level average reduction ratio in offpeak loads, adoption rate 20%,',np.mean((demand_supply_results_ratio['week_offpeak_after']-demand_supply_results_ratio['week_offpeak_before'])/demand_supply_results_ratio['week_offpeak_before']))
        print('ZIP code level average reduction in offpeak loads, adoption rate 20%,',np.mean(demand_supply_results['week_offpeak_after']-demand_supply_results['week_offpeak_before']))

        demand_supply = self.demand.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==100]
        print('ZIP code level average reduction ratio in offpeak loads, adoption rate 100%,',np.mean((demand_supply_results['week_offpeak_after']-demand_supply_results['week_offpeak_before'])/demand_supply_results['week_offpeak_before']))
        print('ZIP code level average reduction in offpeak loads, adoption rate 100%,',np.mean(demand_supply_results['week_offpeak_after']-demand_supply_results['week_offpeak_before']))
    
    def calFutureSupplyTemporal(self):
        print("=====================Figure 5F=======================")
        demand_supply = self.supply.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==20]
        print('Ratio of ZIP Codes experiences insufficient public charging station at peak hours before recommendations, adoption rate 20%',len(demand_supply_results[demand_supply_results['week_peak_before_gap']<0])/len(demand_supply_results))
        print('ZIP Code level average insufficiency, mean:',demand_supply_results['week_peak_before_gap'].mean()/1000,'std:',demand_supply_results['week_peak_before_gap'].std()/1000)

        demand_supply_results = demand_supply[demand_supply['adoption rate']==100]
        print('Ratio of ZIP Codes experiences insufficient public charging station at peak hours before recommendations, adoption rate 100%',len(demand_supply_results[demand_supply_results['week_peak_before_gap']<0])/len(demand_supply_results))
        print('ZIP Code level average insufficiency, mean:',demand_supply_results['week_peak_before_gap'].mean()/1000,',std:',demand_supply_results['week_peak_before_gap'].std()/1000)

        demand_supply_results = demand_supply[demand_supply['adoption rate']==20]
        print('Ratio of ZIP Codes experiences insufficient public charging station at peak hours after recommendations, adoption rate 20%', len(demand_supply_results[demand_supply_results['week_peak_after_gap']<0])/len(demand_supply_results))
        print('ZIP Code level average insufficiency, mean:',demand_supply_results['week_peak_after_gap'].mean()/1000,',std:',demand_supply_results['week_peak_after_gap'].std()/1000)

        demand_supply_results = demand_supply[demand_supply['adoption rate']==100]
        print('Ratio of ZIP Codes experiences insufficient public charging station at peak hours after recommendations, adoption rate 100%', len(demand_supply_results[demand_supply_results['week_peak_after_gap']<0])/len(demand_supply_results))
        print('ZIP Code level average insufficiency, mean:',demand_supply_results['week_peak_after_gap'].mean()/1000,',std:',demand_supply_results['week_peak_after_gap'].std()/1000)

        print("=====================Figure 5G=======================")
        demand_supply_results = demand_supply[demand_supply['adoption rate']==20]
        print('Ratio of ZIP Codes experiences insufficient public charging station at offpeak hours before recommendations, adoption rate 20%',len(demand_supply_results[demand_supply_results['week_offpeak_before_gap']<0])/len(demand_supply_results))
        print('ZIP Code level average insufficiency, mean:',demand_supply_results['week_offpeak_before_gap'].mean()/1000,',std:',demand_supply_results['week_offpeak_before_gap'].std()/1000)

        demand_supply_results = demand_supply[demand_supply['adoption rate']==100]
        print('Ratio of ZIP Codes experiences insufficient public charging station at offpeak hours before recommendations, adoption rate 100%',len(demand_supply_results[demand_supply_results['week_offpeak_before_gap']<0])/len(demand_supply_results))
        print('ZIP Code level average insufficiency, mean:',demand_supply_results['week_offpeak_before_gap'].mean()/1000,',std:',demand_supply_results['week_offpeak_before_gap'].std()/1000)

        demand_supply_results = demand_supply[demand_supply['adoption rate']==20]
        print('Ratio of ZIP Codes experiences insufficient public charging station at offpeak hours after recommendations, adoption rate 20%',len(demand_supply_results[demand_supply_results['week_offpeak_after_gap']<0])/len(demand_supply_results))
        print('ZIP Code level average insufficiency, mean:',demand_supply_results['week_offpeak_after_gap'].mean()/1000,',std:',demand_supply_results['week_offpeak_after_gap'].std()/1000)

        demand_supply_results = demand_supply[demand_supply['adoption rate']==100]
        print('Ratio of ZIP Codes experiences insufficient public charging station at offpeak hours after recommendations, adoption rate 100%', len(demand_supply_results[demand_supply_results['week_offpeak_after_gap']<0])/len(demand_supply_results))
        print('ZIP Code level average insufficiency, mean:',demand_supply_results['week_offpeak_after_gap'].mean()/1000,'std',demand_supply_results['week_offpeak_after_gap'].std()/1000)
    
    def calFutureNetChange(self):
        print("=====================Net Change=======================")
        demand_supply = self.supply.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==20]
        demand_supply_results['demand_before'] = demand_supply_results[['week_peak_before', 'week_offpeak_before']].max(axis=1)
        demand_supply_results['demand_after'] = demand_supply_results[['week_peak_after', 'week_offpeak_after']].max(axis=1)
        print('ZIP Code level average increase in station insufficiency caused by recommendations, adoption rate 20%, power 50 [kW]',(demand_supply_results['demand_after']-demand_supply_results['demand_before']).mean()/50)
        print('ZIP Code level average increase in station insufficiency caused by recommendations, adoption rate 20%, power 6.6 [kW]',(demand_supply_results['demand_after']-demand_supply_results['demand_before']).mean()/6.6)
        demand_supply = self.supply.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==100]
        demand_supply_results['demand_before'] = demand_supply_results[['week_peak_before', 'week_offpeak_before']].max(axis=1)
        demand_supply_results['demand_after'] = demand_supply_results[['week_peak_after', 'week_offpeak_after']].max(axis=1)
        print('ZIP Code level average increase in station insufficiency caused by recommendations, adoption rate 100%, power 50 [kW]',(demand_supply_results['demand_after']-demand_supply_results['demand_before']).mean()/50)
        print('ZIP Code level average increase in station insufficiency caused by recommendations, adoption rate 100%, power 6.6 [kW]',(demand_supply_results['demand_after']-demand_supply_results['demand_before']).mean()/6.6)
    
    def calFutureDemandYear(self,year): 
        print("=====================Future Demand=======================")
        print('Year',year)
        df_user_session = self.df_user_session.copy()
        df_shift_session = self.df_shift_session.copy()
        user_select = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(year) + 'p.pkl'), 'rb'), encoding='bytes')
        df_before = df_user_session[df_user_session['id'].isin(user_select)]
        df_after = df_shift_session[df_shift_session['id'].isin(user_select)]

        df_before = df_before.fillna(-1)
        df_after = df_after.fillna(-1)
        df_before = pd.pivot_table(df_before,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()
        df_after = pd.pivot_table(df_after,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()
        home_load, work_load, other_load, total_load = self.simEVLoad(df_before)
        homes_load, works_load, others_load, totals_load = self.simEVLoad(df_after)

        week_peak_total = total_load.values[:,self.week_peak]
        week_offpeak_total = total_load.values[:,self.week_offpeak]
        week_peak_totals = totals_load.values[:,self.week_peak]
        week_offpeak_totals = totals_load.values[:,self.week_offpeak]

        print('Bay Area reduction in charging needs during on-peak hours',np.max(np.sum(week_peak_total,axis=0))-np.max(np.sum(week_peak_totals,axis=0)))
        print('Bay Area reduction in charging needs during off-peak hours',np.max(np.sum(week_offpeak_total,axis=0))-np.max(np.sum(week_offpeak_totals,axis=0)))
        print('Bay Area peak charging load during on-peak hours',np.max(np.sum(week_peak_total,axis=0)))
        print('Bay Area peak charging load reduction ratio during on-peak hours',(np.max(np.sum(week_peak_total,axis=0))-np.max(np.sum(week_peak_totals,axis=0)))/np.max(np.sum(week_peak_total,axis=0)) )

    def calFutureSupplyYear(self,year): 
        print("=====================Future Supply=======================")
        demand_supply = self.supply.copy()
        demand_supply_results = demand_supply[demand_supply['adoption rate']==year]
        demand_supply_results['demand_before'] = demand_supply_results[['week_peak_before', 'week_offpeak_before']].max(axis=1)
        demand_supply_results['demand_after'] = demand_supply_results[['week_peak_after', 'week_offpeak_after']].max(axis=1)
        power = 50
        print('# Bay Area missing charging stations before recomendations, power 50 [kW]',(demand_supply_results['demand_before']-demand_supply_results['Power']).sum()/power)
        print('# Bay Area missing charging stations after recomendations, power 50 [kW]',(demand_supply_results['demand_after']-demand_supply_results['Power']).sum()/power)
        power = 6.6
        print('# Bay Area missing charging stations before recomendations, power 6.6 [kW]',(demand_supply_results['demand_before']-demand_supply_results['Power']).sum()/power)
        print('# Bay Area missing charging stations after recomendations, power 6.6 [kW]',(demand_supply_results['demand_after']-demand_supply_results['Power']).sum()/power)

if __name__ == "__main__":
    anastat = AnaStat(0.5,0.54,17,21,1,9)
    anastat.calChargingGroup()
    anastat.calDemandTemporal()
    anastat.calDemandSpatial()
    anastat.calShaveTemporal()
    anastat.calShaveSector()
    anastat.calSupplyShave()
    anastat.calFutureDemo()
    anastat.calFutureDemandTemporal()
    anastat.calFutureDemandSpatial()
    anastat.calFutureSupplyTemporal()
    anastat.calFutureNetChange()
    anastat.calFutureDemandYear(0.3)
    anastat.calFutureDemandYear(0.5)
    anastat.calFutureSupplyYear(0.5)
    
    
    