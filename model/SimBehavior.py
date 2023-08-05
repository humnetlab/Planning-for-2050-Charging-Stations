import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def parallelize_dataframe(args, func, n_cores):
    pool = Pool(n_cores)
    results = pool.map(func,args)
    pool.close()
    pool.join()
    return results

def read_week_segment_file():
    p = {}
    for G in range(136):
        p[G] = pd.read_csv(os.path.join('..','data','speech','SPEECh Original Model 136/pz_weekday_g_'+str(G)+'.csv'))
    return p

def read_weekend_segment_file():
    p = {}
    for G in range(136):
        p[G] = pd.read_csv(os.path.join('..','data','speech','SPEECh Original Model 136/pz_weekend_g_'+str(G)+'.csv'))
    return p

def read_week_session_file():
    p = {}
    for G in range(136):
        p[G] = {}
        for z in ['home_l2','mud_l2','work_l2','public_l2','public_l3']:
            key = os.path.join('..','data','speech','SPEECh Original Model 136/GMMs/weekday_'+z+'_'+str(G)+'.p')
            try:
                p[G][z] = pickle.load(open(key, "rb"))
            except:
                continue
    return p

def read_weekend_session_file():
    p = {}
    for G in range(136):
        p[G] = {}
        for z in ['home_l2','mud_l2','work_l2','public_l2','public_l3']:
            key = os.path.join('..','data','speech','SPEECh Original Model 136/GMMs/weekend_'+z+'_'+str(G)+'.p')
            try:
                p[G][z] = pickle.load(open(key, "rb"))
            except:
                continue
    return p 

def getSegment(G,homes,works,levels,days,stays,p_zw,p_ze):
    choice_day = []
    choice_prob = []
    choice_type = []
    
    for stay,day in zip(stays,days):
        if (day != 5) and (day != 6):
            p = p_zw[G]
            if stay == 'h':
                if homes==0:
                    type_list = ['home_l2 - Fraction of weekdays with session','public_l3 - Fraction of weekdays with session','public_l2 - Fraction of weekdays with session']
                else:
                    type_list = ['mud_l2 - Fraction of weekdays with session','public_l3 - Fraction of weekdays with session','public_l2 - Fraction of weekdays with session']
            if stay == 'w':
                if works==0:
                    type_list = ['public_l3 - Fraction of weekdays with session','public_l2 - Fraction of weekdays with session']
                else:
                    type_list = ['work_l2 - Fraction of weekdays with session','public_l3 - Fraction of weekdays with session','public_l2 - Fraction of weekdays with session']
            if stay[0] == 'o':
                type_list = ['public_l3 - Fraction of weekdays with session','public_l2 - Fraction of weekdays with session']
        else:
            p = p_ze[G]
            if stay == 'h':
                if homes==0:
                    type_list = ['home_l2 - Fraction of weekenddays with session','public_l3 - Fraction of weekenddays with session','public_l2 - Fraction of weekenddays with session']
                else:
                    type_list = ['mud_l2 - Fraction of weekenddays with session','public_l3 - Fraction of weekenddays with session','public_l2 - Fraction of weekenddays with session']
            if stay == 'w':
                if works==0:
                    type_list = ['public_l3 - Fraction of weekenddays with session','public_l2 - Fraction of weekenddays with session']
                else:
                    type_list = ['work_l2 - Fraction of weekenddays with session', 'public_l3 - Fraction of weekenddays with session', 'public_l2 - Fraction of weekenddays with session']
            if stay[0] == 'o':
                type_list = ['public_l3 - Fraction of weekenddays with session','public_l2 - Fraction of weekenddays with session']
        
        freq_mean = np.mean(p[type_list].values,axis=0)
        target_type = type_list[np.argmax(freq_mean, axis=0)].split(' ')[0]
        if target_type == 'home_l2' and levels==1:
            target_type = 'home_l1'
        target_prob = np.max(freq_mean)
        target_choice = np.random.choice(2, 1, p=[1-target_prob,target_prob])[0]

        choice_day.append(target_choice)
        choice_type.append(target_type)
        if target_prob==0:
            choice_prob.append(target_prob)
        else: 
            choice_prob.append(1/target_prob)
    return choice_day,choice_prob,choice_type

def getSession(G,zs,ztypes,stays,days,p_sw,p_se):
    energy_day = np.zeros(len(zs))
    for day,stay,z,types,i in zip(days,stays,zs,ztypes,range(len(zs))):
        if types in ['home_l1']:
            types = 'home_l2'
        if z == 1 and i != 0:
            if (day != 5) and (day != 6):
                try:
                    p_s = p_sw[G][types]
                    energy_day[i] = np.clip(p_s.sample(1)[0][0, 1], 0, 100)
                except:
                    if types in ['home_l2']:
                        p_s = p_sw[96][types]
                    elif types in ['mud_l2']:
                        p_s = p_sw[95][types]
                    elif types in ['work_l2']:
                        p_s = p_sw[98][types]
                    elif types in ['public_l2','public_l3']:
                        p_s = p_sw[105][types]
                    energy_day[i] = np.clip(p_s.sample(1)[0][0, 1], 0, 100)
            else:
                try:
                    p_s = p_se[G][types]
                    energy_day[i] = np.clip(p_s.sample(1)[0][0, 1], 0, 100)
                except:
                    if types in ['home_l2']:
                        p_s = p_se[96][types]
                    elif types in ['mud_l2']:
                        p_s = p_se[95][types]
                    elif types in ['work_l2']:
                        p_s = p_se[98][types]
                    elif types in ['public_l2','public_l3']:
                        p_s = p_se[105][types]
                    energy_day[i] = np.clip(p_s.sample(1)[0][0, 1], 0, 100) 
    return np.array(energy_day)

def calSession(para):
    batch_id,batch_size,batch_name,G,U,H,W,L = para
    total_batch = range(len(G))
    user_select_batch = total_batch[batch_id*np.ceil(batch_size).astype(int):(batch_id+1)*np.ceil(batch_size).astype(int)]
    userTraj_session = {}; userTraj_type = {}
    for i in user_select_batch:
        stays = userTraj_label[U[i]]
        days = (np.array(userTraj_time[U[i]])/144).astype(int)
        z_s,z_prob,z_type = getSegment(G[i],H[i],W[i],L[i],days,stays,p_zw,p_ze)
        simulate = getSession(G[i],z_s,z_type,stays,days,p_sw,p_se)
        userTraj_session[U[i]] = simulate
        userTraj_type[U[i]] = z_type
    
    pickle.dump(userTraj_session, open(os.path.join(batch_name,'userTraj_session_'+str(batch_id)+'.pkl'),'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(userTraj_type, open(os.path.join(batch_name,'userTraj_type_'+str(batch_id)+'.pkl'),'wb'), pickle.HIGHEST_PROTOCOL)

    del userTraj_session
    del userTraj_type

p_zw = read_week_segment_file(); p_ze = read_weekend_segment_file()
p_sw = read_week_session_file(); p_se = read_weekend_session_file()
userTraj_label = pickle.load(open(os.path.join('..','result','mobility','userTraj_label_week.pkl'), 'rb'), encoding='bytes')
userTraj_zipcode = pickle.load(open(os.path.join('..','result','mobility','userTraj_zipcode_week.pkl'), 'rb'), encoding='bytes')
userTraj_time = pickle.load(open(os.path.join('..','result','mobility','userTraj_time_week.pkl'), 'rb'), encoding='bytes')
userTraj_route_energy = pickle.load(open(os.path.join('..','result','mobility','userTraj_route_energy.pkl'), 'rb'), encoding='bytes')
userTraj_route_time = pickle.load(open(os.path.join('..','result','mobility','userTraj_route_time.pkl'), 'rb'), encoding='bytes')

class SimBehavior():
    def __init__(self, work_l2_access,home_l2_access):
        super().__init__()
        self.work_l2_access = work_l2_access
        self.home_l2_access = home_l2_access
        self.mobility_folder_name = os.path.join('..','result','mobility')
        self.adopter_folder_name = os.path.join('..','result','adopter','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access))
        self.behavior_folder_name = os.path.join('..','result','behavior','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access))
        if not os.path.exists(self.behavior_folder_name):
            os.makedirs(self.behavior_folder_name)

    def read_group_file(self):
        p = {}
        for A in range(4):
            if A == 0:
                p[A] = pd.read_csv(os.path.join('..','data','speech','pgandsamplecode136/pg_HnoWno_n136.csv'),index_col=0)
            if A == 1:
                p[A] = pd.read_csv(os.path.join('..','data','speech','pgandsamplecode136/pg_HyesWno_n136.csv'),index_col=0)
            if A == 2:
                p[A] = pd.read_csv(os.path.join('..','data','speech','pgandsamplecode136/pg_HnoWyes_n136.csv'),index_col=0)
            if A == 3:
                p[A] = pd.read_csv(os.path.join('..','data','speech','pgandsamplecode136/pg_HyesWyes_n136.csv'),index_col=0)
        return p

    def getGroup(self,A,D,p):
        p_a = p[A]
        prob = p_a[str(D)].values
        G = int(np.random.choice(136, 1, p=prob))
        return G  
    
    def calGroup(self):
        df_user_inf = pickle.load(open(os.path.join(self.adopter_folder_name,'df_user_inf.pkl'), 'rb'), encoding='bytes')
        df_user_inf['personConsumptionBin'] = np.clip((df_user_inf['personConsumption']*365/250).astype(int)*250, 0, 4750)
        A = df_user_inf['personAccess'].values
        D = df_user_inf['personConsumptionBin'].values

        p_g = self.read_group_file()
        G = np.zeros(len(A))
        for i in range(len(A)):
            G[i] = int(self.getGroup(A[i],D[i],p_g))
        df_user_inf['personGroup']=G.astype(int)

        df_user_inf.to_csv(os.path.join(self.behavior_folder_name,'df_user_group.csv'),index=False)

    def getPercentage(self,group,rate):
        group_zipcode = pd.pivot_table(group, values='userID', index ='new_label', columns ='homeZipcode',aggfunc=np.count_nonzero).sum()
        group_zipcode_percentage = (pd.pivot_table(group, values='userID', index ='new_label', columns ='homeZipcode',aggfunc=np.count_nonzero)/group_zipcode).T*100
        group_zipcode_percentage['adoption_rate'] = rate
        return group_zipcode_percentage

    def calBehavior(self):
        df_user_inf = pd.read_csv(os.path.join(self.behavior_folder_name,'df_user_group.csv'))
        G = df_user_inf['personGroup'].values
        U = df_user_inf['userID'].values
        H = df_user_inf['personHouse'].values
        W = df_user_inf['personAccess_w'].values
        L = df_user_inf['personAccess_hl1'].values
        user_num = len(G); run_core = 24; batch_size = user_num/run_core

        results = parallelize_dataframe([[batch_id,batch_size,self.behavior_folder_name,G,U,H,W,L]for batch_id in range(run_core)],calSession, n_cores=run_core)
        userTraj_session = {}
        for batch_id in range(24):
            userTraj_session.update(pickle.load(open(os.path.join(self.behavior_folder_name,'userTraj_session_'+str(batch_id)+'.pkl'), 'rb'), encoding='bytes'))
            os.remove(os.path.join(self.behavior_folder_name,'userTraj_session_'+str(batch_id)+'.pkl'))
        userTraj_type = {}
        for batch_id in range(24):
            userTraj_type.update(pickle.load(open(os.path.join(self.behavior_folder_name,'userTraj_type_'+str(batch_id)+'.pkl'), 'rb'), encoding='bytes'))
            os.remove(os.path.join(self.behavior_folder_name,'userTraj_type_'+str(batch_id)+'.pkl'))
            
        tes = 21077; nis = 11470; session_rate = {'home_l1':1.2, 'home_l2':6.6, 'mud_l2':6.6, 'work_l2':6.6, 'public_l2':6.6, 'public_l3':50}
        prob_sum = nis+tes
        type_prob = [tes/prob_sum, nis/prob_sum] 

        user_id = []; stay_type = []; stay_zipcode = []; arrive_time = []; depature_time = []; session_energy = []; session_type = []
        userTraj_session_timegeo = {}; userTraj_energy_timegeo = {}; userTraj_soc_timegeo={}; userTraj_capacity_timegeo={}
        
        for id in list(U): 
            energy = np.concatenate([np.array(userTraj_route_energy[id]),np.array([0])]).copy()
            car_type = np.random.choice(2, 1, p = type_prob)[0]

            if car_type == 0:
                capacity = 82
                factor = 0.25/0.3
                energy = energy*factor
            if car_type == 1:
                capacity = 40
                factor = 1
                energy = energy*factor
                
            split = (userTraj_session[id])/(sum(userTraj_session[id])+1e-5)
            session = split*np.sum(energy)

            capacity_upper = 1; capacity_lower = 0.2; delta = []; 
            soc_start = np.clip(np.random.normal(capacity*0.8, capacity*0.2, 1)[0],capacity_lower*capacity,capacity_upper*capacity)

            travel_time = np.ceil(userTraj_route_time[id]*6)
            for i in range(len(session)):
                try:
                    dtime = userTraj_time[id][i+1]
                    dept = userTraj_time[id][i+1]
                except:
                    dtime = 144*7
                    dept = 144*7
                if i>0:
                    arvt = min(userTraj_time[id][i]+travel_time[i-1],dtime)
                else:
                    arvt = 1
                session[i] = min(session[i],(dept-arvt)/6*session_rate[userTraj_type[id][i]])

                cum = soc_start+sum(session[:i+1])-sum(energy[:i+1])
                if cum>capacity*capacity_upper:
                    session[i] = session[i] - (cum-capacity*capacity_upper)
                if cum<capacity*capacity_lower:
                    energy[i] = energy[i]-(capacity*capacity_lower-cum)
                delta.append(session[i])
                delta.append(-energy[i])
            soc_net = np.cumsum(delta)

            userTraj_session_timegeo[id] = session
            userTraj_energy_timegeo[id] = energy
            userTraj_soc_timegeo[id] = soc_start+soc_net
            userTraj_capacity_timegeo[id] = capacity

            for i in range(len(userTraj_label[id])):
                try:
                    user_id.append(id)
                    session_energy.append(session[i])
                    stay_type.append(userTraj_label[id][i])
                    stay_zipcode.append(userTraj_zipcode[id][i])
                    session_type.append(userTraj_type[id][i])
                    try:
                        dtime = userTraj_time[id][i+1]
                        depature_time.append(userTraj_time[id][i+1])
                    except:
                        dtime = 144*7
                        depature_time.append(144*7)
                    if i>0:
                        arrive_time.append(min(userTraj_time[id][i]+travel_time[i-1],dtime))
                    else:
                        arrive_time.append(1)
                except:
                    continue

        pickle.dump(userTraj_session_timegeo, open(os.path.join(self.behavior_folder_name,'userTraj_session_timegeo.pkl'),'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(userTraj_energy_timegeo, open(os.path.join(self.behavior_folder_name,'userTraj_energy_timegeo.pkl'),'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(userTraj_soc_timegeo, open(os.path.join(self.behavior_folder_name,'userTraj_soc_timegeo.pkl'),'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(userTraj_capacity_timegeo, open(os.path.join(self.behavior_folder_name,'userTraj_capacity_timegeo.pkl'),'wb'), pickle.HIGHEST_PROTOCOL)

        df = pd.DataFrame(data={'id': user_id,'arrive_time': arrive_time, 'depature_time':depature_time, 'session_energy':session_energy, 'stay_type':stay_type, 'stay_zipcode':stay_zipcode, 'session_type': session_type}) 
        df = df.merge(df_user_inf[['userID','personAccess_hl1','personAccess_hl2']],right_on='userID',left_on='id')	
        df.to_csv(os.path.join(self.behavior_folder_name,'simulated_session.csv'),index = False)
        del df

if __name__ == "__main__":
    smbhr = SimBehavior(0.50,0.54)
    smbhr.calGroup()
    smbhr.calBehavior()
