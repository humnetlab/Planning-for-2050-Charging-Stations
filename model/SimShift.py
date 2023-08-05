import pickle
from multiprocessing import Pool
import numpy as np
import geopandas as gpd
import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

def parallelize_dataframe(args, func, n_cores):
    pool = Pool(n_cores)
    results = pool.map(func,args)
    pool.close()
    pool.join()
    return results

def calDemand(para):
    batch_id,batch_name,week_peak,max_stay_list,df_total,user_select_batch,userTraj_soc_timegeo,userTraj_capacity_timegeo = para
    capacity_upper = 1; capacity_lower = 0.2
    session_rate = {'home_l1':1.2, 'home_l2':6.6, 'mud_l2':6.6, 'work_l2':6.6, 'public_l2':6.6, 'public_l3':50}
    list_of_df = []

    for id in user_select_batch:
        df = df_total[(df_total['id']==id)].copy()
        arrive_time = df['arrive_time'].values.copy()
        departure_time = df['depature_time'].values.copy()
        session_type = df['session_type'].values.copy()
        session = df['session_energy'].values.copy()
        
        out_id = df['id'].values.copy()
        out_arrive_time = df['arrive_time'].values.copy()
        out_departure_time= df['depature_time'].values.copy()
        out_session_energy = df['session_energy'].values.copy()
        out_session_type = df['session_type'].values.copy()
        out_stay_zipcode = df['stay_zipcode'].values.copy()
        out_is_peak=np.zeros(len(arrive_time))
        out_is_shift=np.zeros(len(arrive_time))

        label = np.array(userTraj_label[id]).copy()
        stay_zipcode = np.array(userTraj_zipcode[id]).copy()
        userTraj_soc = userTraj_soc_timegeo[id]
        capacity = userTraj_capacity_timegeo[id]
        n_places = len(label)

        for i in range(n_places):
            b_session_start = int(arrive_time[i])
            b_session_end = int(arrive_time[i]+out_session_energy[i]/session_rate[out_session_type[i]]*6+1)
            b_session_energy = out_session_energy[i]

            if (out_session_energy[i]>0) and (set(range(b_session_start,b_session_end)) & set(week_peak)):
                out_is_peak[i] = 1
                for k in max_stay_list: 
                    j = i+k
                    if j<n_places and j>=0:
                        a_session_start = int(arrive_time[j])
                        a_session_end = int(arrive_time[j]+(session[j]+b_session_energy)/session_rate[session_type[j]]*6+1) 
                        a_session_departure = departure_time[j]
                        a_soc = userTraj_soc.copy() #userTraj_soc_start[id]+np.cumsum(session)[:max(i,j)+1]-userTraj_discharging[id][:max(i,j)+1]
                        a_soc[2*i:] = a_soc[2*i:]-b_session_energy
                        a_soc[2*j:] = a_soc[2*j:]+b_session_energy

                        if ((set(range(a_session_start,a_session_end)) & set(week_peak))==set()) and \
                        (a_session_end<=a_session_departure) and \
                        (min(a_soc)>=capacity_lower*capacity) and (max(a_soc)<=capacity*capacity_upper):
                            session[j] = session[j]+b_session_energy; session[i] = session[i]-b_session_energy
                            userTraj_soc[2*j:] = userTraj_soc[2*j:]+b_session_energy
                            userTraj_soc[2*i:] = userTraj_soc[2*i:]-b_session_energy

                            out_arrive_time[i] = arrive_time[j]
                            out_departure_time[i] = departure_time[j]
                            out_session_type[i] = session_type[j]
                            try:
                                out_stay_zipcode[i] = stay_zipcode[j]
                            except:
                                out_stay_zipcode[i] = np.nan

                            out_is_shift[i] = k
                            break
                    
        df_id = pd.DataFrame({'id':out_id, 'arrive_time':out_arrive_time, \
            'depature_time':out_departure_time, 'session_energy': out_session_energy,\
             'stay_zipcode':out_stay_zipcode, 'session_type':out_session_type, 'original_session_type':session_type, \
             'original_arrive_time':arrive_time, 'original_depature_time':departure_time, 'original_stay_zipcode': stay_zipcode,\
             'is_shift': out_is_shift})
        list_of_df.append(df_id)
    df_concat = pd.concat(list_of_df)
    pickle.dump(df_concat, open(os.path.join(batch_name,'userTraj_shift_'+str(batch_id)+'.pkl'),'wb'), pickle.HIGHEST_PROTOCOL)
    del df_concat,list_of_df

userTraj_label = pickle.load(open(os.path.join('..','result','mobility','userTraj_label_week.pkl'), 'rb'), encoding='bytes')
userTraj_time = pickle.load(open(os.path.join('..','result','mobility','userTraj_time_week.pkl'), 'rb'), encoding='bytes')
userTraj_zipcode = pickle.load(open(os.path.join('..','result','mobility','userTraj_zipcode_week.pkl'), 'rb'), encoding='bytes')


class SimShift():
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
        if not os.path.exists(self.shift_folder_name):
            os.makedirs(self.shift_folder_name)

    def calPeakHour(self):
        peak_start = self.peak_start; peak_end=self.peak_end
        week_peak = []; week_offpeak = []
        for hour in range(168):
            if hour<5*24:
                if ((hour)%24 >= peak_start) and ((hour)%24 < peak_end):
                    week_peak.append(hour)
                else:
                    week_offpeak.append(hour)
            else:
                week_offpeak.append(hour)
        self.week_peak_hour = week_peak; self.week_offpeak_hour = week_offpeak
        return week_peak,week_offpeak
    
    def calPeakMin(self):
        peak_start = self.peak_start; peak_end=self.peak_end
        week_peak = []; week_offpeak = []
        for hour in range(168*6):
            if hour<5*24*6:
                if ((hour/6)%24 >= peak_start) and ((hour/6)%24 < peak_end):
                    week_peak.append(hour)
                else:
                    week_offpeak.append(hour)
            else:
                week_offpeak.append(hour)
        self.week_peak_min = week_peak; self.week_offpeak_min = week_offpeak
        return week_peak,week_offpeak
    
    def calMaxStay(self):
        tolerable_list = []
        for i in range(self.max_stay):
            tolerable_list.append(i)
            tolerable_list.append(-i)
        self.max_stay_list = tolerable_list

    def calZIPCode(self):
        df_user_session = pd.read_csv(os.path.join(self.behavior_folder_name,'simulated_session.csv'))
        self.zipcode_list = df_user_session['stay_zipcode'].dropna().unique()

    def calExistedStation(self):
        info = pd.read_csv(os.path.join('..','data','supply','usage_type.csv'))
        geocs = gpd.read_file(os.path.join('..','data','supply','evcs.geojson'))
        geozip = gpd.read_file(os.path.join('..','data','census','sfbay_zip.geojson')) 
        
        geocs['UsageTypeID'] = geocs['UsageTypeID'].astype(int)
        geocs = geocs.merge(info,left_on='UsageTypeID',right_on='ID')
        evcs_sfbay = gpd.sjoin(geozip,geocs,how="inner")
        evcs_sfbay['Power'] = evcs_sfbay['PowerKW'] * evcs_sfbay['Quantity']
        evcs_sfbay = evcs_sfbay[evcs_sfbay['StationType']=='Public Charging Places']
        evcs_sfbay_power = pd.pivot_table(evcs_sfbay, values='Power', index=['ZCTA5CE10'],aggfunc=np.sum)
        self.evcs_sfbay_power = evcs_sfbay_power
    
    def calPre(self):
        self.calPeakHour()
        self.calPeakMin()
        self.calZIPCode()
        self.calMaxStay()
        self.calExistedStation()
    
    def calShift(self):
        self.calPre()
        df_user_session = pd.read_csv(os.path.join(self.behavior_folder_name,'simulated_session.csv'))
        userTraj_soc_timegeo = pickle.load(open(os.path.join(self.behavior_folder_name,'userTraj_soc_timegeo.pkl'), 'rb'), encoding='bytes')
        userTraj_capacity_timegeo = pickle.load(open(os.path.join(self.behavior_folder_name,'userTraj_capacity_timegeo.pkl'), 'rb'), encoding='bytes')

        chunklen = np.ceil(len(userTraj_soc_timegeo.keys())/48).astype(int)
        dictlist = list(userTraj_soc_timegeo.items())
        chunked_soc = [ dict(dictlist[i:i + chunklen]) for i in range(0, len(dictlist), chunklen) ]
        dictlist = list(userTraj_capacity_timegeo.items())
        chunked_capacity = [ dict(dictlist[i:i + chunklen]) for i in range(0, len(dictlist), chunklen) ]

        results = parallelize_dataframe([[batch_id,self.shift_folder_name,self.week_peak_min,self.max_stay_list,df_user_session[df_user_session['userID'].isin(list(chunked_capacity[batch_id].keys()))].copy(),list(chunked_capacity[batch_id].keys()),chunked_soc[batch_id],chunked_capacity[batch_id]] for batch_id in range(24)],calDemand, n_cores=24)
        results = parallelize_dataframe([[batch_id,self.shift_folder_name,self.week_peak_min,self.max_stay_list,df_user_session[df_user_session['userID'].isin(list(chunked_capacity[batch_id].keys()))].copy(),list(chunked_capacity[batch_id].keys()),chunked_soc[batch_id],chunked_capacity[batch_id]] for batch_id in range(24,48)],calDemand, n_cores=24)
        df_chunks = [] 
        for batch_id in range(48):
            df_chunks.append(pickle.load(open(os.path.join(self.shift_folder_name,'userTraj_shift_'+str(batch_id)+'.pkl'), 'rb'), encoding='bytes'))
            os.remove(os.path.join(self.shift_folder_name,'userTraj_shift_'+str(batch_id)+'.pkl'))
        df_shift_session = pd.concat(df_chunks)
        df_shift_session.to_csv(os.path.join(self.shift_folder_name,'shifted_session.csv'),index = False)    
        del df_shift_session

    def calAcceptance(self):
        df_shift_session = pd.read_csv(os.path.join(self.shift_folder_name,'shifted_session.csv'))
        user_follow = pd.DataFrame([])
        user_follow['id'] = df_shift_session.id.unique()
        user_follow['is_follow'] = np.random.choice([0,1], df_shift_session.id.nunique(), p=[1-self.acceptance,self.acceptance])
        df_shift_session_follow = df_shift_session.merge(user_follow,left_on='id',right_on='id')

        df_shift_session_follow_yes = pd.DataFrame([])
        df_shift_session_follow_yes['id'] = df_shift_session_follow[df_shift_session_follow['is_follow']==1]['id'].values
        df_shift_session_follow_yes['arrive_time'] = df_shift_session_follow[df_shift_session_follow['is_follow']==1]['arrive_time'].values
        df_shift_session_follow_yes['depature_time'] = df_shift_session_follow[df_shift_session_follow['is_follow']==1]['depature_time'].values
        df_shift_session_follow_yes['session_energy'] = df_shift_session_follow[df_shift_session_follow['is_follow']==1]['session_energy'].values
        df_shift_session_follow_yes['session_type'] = df_shift_session_follow[df_shift_session_follow['is_follow']==1]['session_type'].values
        df_shift_session_follow_yes['stay_zipcode'] = df_shift_session_follow[df_shift_session_follow['is_follow']==1]['stay_zipcode'].values
        df_shift_session_follow_yes['is_shift'] = df_shift_session_follow[df_shift_session_follow['is_follow']==1]['is_shift'].values
        df_shift_session_follow_yes['original_session_type'] = df_shift_session_follow[df_shift_session_follow['is_follow']==1]['original_session_type'].values

        df_shift_session_follow_no = pd.DataFrame([])
        df_shift_session_follow_no['id'] = df_shift_session_follow[df_shift_session_follow['is_follow']==0]['id'].values
        df_shift_session_follow_no['arrive_time'] = df_shift_session_follow[df_shift_session_follow['is_follow']==0]['original_arrive_time'].values
        df_shift_session_follow_no['depature_time'] = df_shift_session_follow[df_shift_session_follow['is_follow']==0]['original_depature_time'].values
        df_shift_session_follow_no['session_energy'] = df_shift_session_follow[df_shift_session_follow['is_follow']==0]['session_energy'].values
        df_shift_session_follow_no['session_type'] = df_shift_session_follow[df_shift_session_follow['is_follow']==0]['original_session_type'].values
        df_shift_session_follow_no['stay_zipcode'] = df_shift_session_follow[df_shift_session_follow['is_follow']==0]['original_stay_zipcode'].values
        df_shift_session_follow_no['is_shift'] = df_shift_session_follow[df_shift_session_follow['is_follow']==0]['is_shift'].values
        df_shift_session_follow_no['original_session_type'] = df_shift_session_follow[df_shift_session_follow['is_follow']==0]['original_session_type'].values
        
        df_shift_session_acceptance = pd.concat([df_shift_session_follow_yes,df_shift_session_follow_no])
        df_shift_session_acceptance.to_csv(os.path.join(self.shift_folder_name,'shifted_session.csv'),index=False)
        del df_shift_session, df_shift_session_acceptance, df_shift_session_follow_yes,df_shift_session_follow_no

    def simEVLoad(self,df):
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

        for zipcode in self.zipcode_list:
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

    def calResults(self):
        df_user_session = pd.read_csv(os.path.join(self.behavior_folder_name,'simulated_session.csv'))
        df_shift_session = pd.read_csv(os.path.join(self.shift_folder_name,'shifted_session.csv')) 

        demand_list = []
        for (year,rate) in zip([0,2,3,4,5,6],[2,20,40,60,80,100]):

            user_select = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(year) + 'p.pkl'), 'rb'), encoding='bytes')
            df_before = df_user_session[df_user_session['id'].isin(user_select)]
            df_after = df_shift_session[df_shift_session['id'].isin(user_select)]
            df_before = df_before.fillna(-1)
            df_after = df_after.fillna(-1)
            df_before = pd.pivot_table(df_before,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()
            df_after = pd.pivot_table(df_after,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()

            home_load, work_load, other_load, total_load = self.simEVLoad(df_before)
            homes_load, works_load, others_load, totals_load = self.simEVLoad(df_after)

            total = total_load
            totals = totals_load

            week_peak_total = total.values[:,self.week_peak_hour]
            week_offpeak_total = total.values[:,self.week_offpeak_hour]

            week_peak_totals = totals.values[:,self.week_peak_hour]
            week_offpeak_totals = totals.values[:,self.week_offpeak_hour]

            demand_results = pd.DataFrame({'zipcode':total.index,'week_peak_before':np.max(week_peak_total,axis=1),'week_offpeak_before':np.max(week_offpeak_total,axis=1),\
                    'week_peak_after':np.max(week_peak_totals,axis=1),'week_offpeak_after':np.max(week_offpeak_totals,axis=1)})
            demand_results = demand_results.loc[demand_results.zipcode.notna()]
            demand_results['zipcode'] = demand_results['zipcode'].astype(int).astype(str)
            demand_supply_results = demand_results.merge(self.evcs_sfbay_power, left_on ='zipcode', right_on='ZCTA5CE10')

            demand_supply_results['adoption rate'] = rate
            demand_list.append(demand_supply_results)
            
        demand = pd.concat(demand_list)
        demand.to_csv(os.path.join(self.shift_folder_name,'demand.csv'),index=False)
        del demand,demand_list

        supply_list = []
        for (year,rate) in zip([0,2,3,4,5,6],[2,20,40,60,80,100]):
            user_select = pickle.load(open(os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(year) + 'p.pkl'), 'rb'), encoding='bytes')
            
            df_before = df_user_session[df_user_session['id'].isin(user_select)]
            df_after = df_shift_session[df_shift_session['id'].isin(user_select)]
            df_before = df_before.fillna(-1)
            df_after = df_after.fillna(-1)
            df_before = pd.pivot_table(df_before,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()
            df_after = pd.pivot_table(df_after,index=['id','arrive_time','depature_time','stay_zipcode','session_type'],values='session_energy',aggfunc=np.sum).reset_index()
            
            home_load, work_load, other_load, total_load = self.simEVLoad(df_before)
            homes_load, works_load, others_load, totals_load = self.simEVLoad(df_after)

            total = other_load
            totals = others_load

            week_peak_total = total.values[:,self.week_peak_hour]
            week_offpeak_total = total.values[:,self.week_offpeak_hour]

            week_peak_totals = totals.values[:,self.week_peak_hour]
            week_offpeak_totals = totals.values[:,self.week_offpeak_hour]

            demand_results = pd.DataFrame({'zipcode':total.index,'week_peak_before':np.max(week_peak_total,axis=1),'week_offpeak_before':np.max(week_offpeak_total,axis=1),\
                    'week_peak_after':np.max(week_peak_totals,axis=1),'week_offpeak_after':np.max(week_offpeak_totals,axis=1)})
            demand_results = demand_results.loc[demand_results.zipcode.notna()]
            demand_results['zipcode'] = demand_results['zipcode'].astype(int).astype(str)
            demand_supply_results = demand_results.merge(self.evcs_sfbay_power, left_on ='zipcode', right_on='ZCTA5CE10')

            demand_supply_results['week_peak_before_gap'] = demand_supply_results['Power']-demand_supply_results['week_peak_before']
            demand_supply_results['week_offpeak_before_gap'] = demand_supply_results['Power']-demand_supply_results['week_offpeak_before']

            demand_supply_results['week_peak_after_gap'] = demand_supply_results['Power']-demand_supply_results['week_peak_after']
            demand_supply_results['week_offpeak_after_gap'] = demand_supply_results['Power']-demand_supply_results['week_offpeak_after']
            demand_supply_results['adoption rate'] = rate
            supply_list.append(demand_supply_results)
        supply = pd.concat(supply_list)
        supply.to_csv(os.path.join(self.shift_folder_name,'supply.csv'),index=False)
        del supply,supply_list

if __name__ == "__main__":
    smshf = SimShift(0.5,0.54,17,21,1,9)
    smshf.calShift()
    smshf.calAcceptance()
    smshf.calResults()

