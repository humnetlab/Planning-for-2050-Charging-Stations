import pickle
import numpy as np
import pandas as pd
import osmnx as ox
from compress_pickle import dump as cdump
from compress_pickle import load as cload
import networkx as nx
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

def speed2consumption(speed_to_consumption,speed,miles):
    if speed < 1:
        speed = 1
    return speed_to_consumption[speed]*miles

def graph2length(graph,s,t):
    return graph[s][t][0]['length']*0.000621371

def graph2time(graph,s,t):
    return graph[s][t][0]['avg_travel_time_sec']/3600

def calEnergy(hour):
    folder_name = os.path.join('..','result','mobility')
    speed_to_consumption = pickle.load(open(os.path.join(folder_name,'speed_to_consumption.pkl'), 'rb'), encoding='bytes')
    nxg = pickle.load(open(os.path.join('..','data','network_uber','network_hour_'+str(hour)+'.pkl'), 'rb'), encoding='bytes')
    GNSp = nx.relabel.convert_node_labels_to_integers(nxg)
    od_route_hour = cload(os.path.join(folder_name,'od_route_hour_'+str(hour)+'.lzma'), compression='lzma')
    route_time = []; route_dis = []; route_energy = []

    for route in od_route_hour:
        time_total = 0; distance_total = 0; energy_total=0
        for n in range(len(route)-1):
            gdis = graph2length(GNSp,route[n],route[n+1])
            gtime = graph2time(GNSp,route[n],route[n+1])
            genergy = speed2consumption(speed_to_consumption,int(gdis/gtime),gdis)
            
            time_total = time_total + gtime
            distance_total = distance_total + gdis
            energy_total = energy_total+ genergy

        route_time.append(time_total)
        route_dis.append(distance_total)
        route_energy.append(energy_total)

    pickle.dump(route_time, open(os.path.join(folder_name,'od_route_time_hour_'+str(hour)+'.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(route_dis, open(os.path.join(folder_name,'od_route_dis_hour_'+str(hour)+'.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(route_energy, open(os.path.join(folder_name,'od_route_energy_hour_'+str(hour)+'.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    

class SimMobility():
    def __init__(self):
        super().__init__()
        self.folder_name = os.path.join('..','result','mobility')
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        
    def calTrajectories(self):
        userTraj_point = {}; userTraj_label = {}; userTraj_zipcode = {}; userTraj_timeSeg = {}; userTraj_time = {}
        pointsInZipcode = pickle.load(open(os.path.join('..','data','census','pointsInZipcode.pkl'), 'rb'))
        timegeo_week_path = os.path.join('..','data','timegeo_week','')
        simFiles = [timegeo_week_path + f for f in os.listdir(timegeo_week_path) if os.path.isfile(os.path.join(timegeo_week_path, f)) and 'txt' in f]
        simFiles = sorted(simFiles)

        userCount = 0
        for f in simFiles:
            data = open(f, 'r')
            for line in data:
                line = line.strip().split(' ')
                if len(line) == 1:
                    userCount += 1
                    if userCount%10000 == 0:
                        print("user : ", userCount)
                    perID = int(line[0].split('-')[0])
                    otherLocations = {}
                    if perID not in userTraj_point:
                        userTraj_point[perID] = []
                        userTraj_label[perID] = []
                        userTraj_zipcode[perID] = []
                        userTraj_timeSeg[perID] = []
                        userTraj_time[perID] = []
                else:
                    timestep = int(line[0])
                    stay_label = str(line[1])
                    lon = float(line[2])
                    lat = float(line[3])
                    if stay_label == 'o':
                        if (lon, lat) not in otherLocations:
                            stay_label = stay_label + str(len(otherLocations))  
                            otherLocations[(lon, lat)] = stay_label
                        else:
                            stay_label = otherLocations[(lon, lat)]
                    try:
                        zipcode = pointsInZipcode[(lon, lat)]
                    except:
                        zipcode = ''
                    userTraj_label[perID].append(stay_label)
                    userTraj_point[perID].append([lon, lat])
                    userTraj_zipcode[perID].append(zipcode)
                    userTraj_time[perID].append(timestep)

        pickle.dump(userTraj_point, open(os.path.join(self.folder_name,'userTraj_point_week.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(userTraj_label, open(os.path.join(self.folder_name,'userTraj_label_week.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(userTraj_zipcode, open(os.path.join(self.folder_name,'userTraj_zipcode_week.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(userTraj_time, open(os.path.join(self.folder_name,'userTraj_time_week.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    
    def calNodes(self):
        traj_point = pickle.load(open(os.path.join(self.folder_name,'userTraj_point_week.pkl'), 'rb'), encoding='bytes')
        traj_time = pickle.load(open(os.path.join(self.folder_name,'userTraj_time_week.pkl'), 'rb'), encoding='bytes')
        od_time = []; user_id = []
        user_keys = list(traj_point.keys())
        for key in user_keys:
            od_time += traj_time[key][1:]
            user_id += [key]*(len(traj_point[key])-1)
        od_time = np.array(od_time).reshape((-1,1))
        user_id = np.array(user_id).reshape((-1,1))
        od_lonlat = np.concatenate([np.hstack([traj_point[key][:-1],traj_point[key][1:]]).reshape((-1,4)) for key in user_keys])
        od_matrix = np.concatenate([user_id,od_time,od_lonlat],axis=1)

        point_lonlat = pd.DataFrame(np.concatenate([od_matrix[:,2:4], od_matrix[:,4:6]]))
        point_lonlat = point_lonlat.drop_duplicates(subset=[0,1]).rename(columns={0:'lon',1:'lat'})
        lon = point_lonlat['lon'].values; lat = point_lonlat['lat'].values
        nxg = nx.relabel.convert_node_labels_to_integers(pickle.load(open(os.path.join('..','data','network_uber','network_hour_7.pkl'), 'rb'), encoding='bytes'))
        near_point = ox.distance.nearest_nodes(nxg, lon, lat, return_dist=False)
        point_lonlat['node_id'] = near_point

        od_point = pd.DataFrame(od_matrix).rename(columns={0:'user_id',1:'od_time',2:'o_lon',3:'o_lat',4:'d_lon',5:'d_lat'})
        od_point = od_point.merge(point_lonlat, left_on=['o_lon','o_lat'], right_on=['lon','lat']).rename(columns={'node_id':'o_node_id'})
        od_point = od_point.merge(point_lonlat, left_on=['d_lon','d_lat'], right_on=['lon','lat']).rename(columns={'node_id':'d_node_id'})
        od_point = od_point[['user_id', 'od_time', 'o_lon', 'o_lat', 'd_lon', 'd_lat','o_node_id','d_node_id']]
        od_point.to_csv(os.path.join(self.folder_name,'od_point.csv'))

    def calRoutes(self):
        od_route = pd.read_csv(os.path.join(self.folder_name,'od_point.csv'),index_col=0)
        od_route['od_time_hour'] = (od_route['od_time'].values/(6.0*7)).astype(int)
        od_route = od_route.drop_duplicates(subset=['o_node_id','d_node_id','od_time_hour'])
        od_route = od_route[['od_time_hour','o_node_id','d_node_id']]
        
        for hour in range(25):
            nxg = pickle.load(open(os.path.join('..','data','network_uber','network_hour_'+str(hour)+'.pkl'), 'rb'), encoding='bytes')
            G_nx = nx.relabel.convert_node_labels_to_integers(nxg)
            od_route_hour = od_route[od_route['od_time_hour']==hour]
            s = list(od_route_hour['o_node_id'].values)
            t = list(od_route_hour['d_node_id'].values)
            od_route_detail = ox.distance.shortest_path(G_nx, s, t, weight='avg_travel_time_sec', cpus=24)
            cdump(od_route_detail, os.path.join(self.folder_name,'od_route_hour_'+str(hour)+'.lzma'), compression='lzma')

    def calEnergies(self):
        self.calNodes()
        self.calRoutes()
        results = parallelize_dataframe([hour for hour in range(25)],calEnergy, n_cores=25)
        od_route = pd.read_csv(os.path.join(self.folder_name,'od_point.csv'),index_col=0)
        od_route['od_time_hour'] = (od_route['od_time'].values/(6.0*7)).astype(int)
        od_route_copy = od_route.copy()
        od_route = od_route.drop_duplicates(subset=['o_node_id','d_node_id','od_time_hour'])

        route_energy_oid = np.concatenate([od_route[od_route['od_time_hour']==hour]['o_node_id'] for hour in range(25)])
        route_energy_did = np.concatenate([od_route[od_route['od_time_hour']==hour]['d_node_id'] for hour in range(25)])
        route_energy_time = np.concatenate([od_route[od_route['od_time_hour']==hour]['od_time_hour'] for hour in range(25)])

        route_energy_total = []; route_time_total = []; route_dis_total = []
        for hour in range(25):
            route_energy = pickle.load(open(os.path.join(self.folder_name,'od_route_energy_hour_'+str(hour)+'.pkl'), 'rb'), encoding='bytes')
            route_energy_total += route_energy
            route_dis = pickle.load(open(os.path.join(self.folder_name,'od_route_dis_hour_'+str(hour)+'.pkl'), 'rb'), encoding='bytes')
            route_dis_total += route_dis
            route_time = pickle.load(open(os.path.join(self.folder_name,'od_route_time_hour_'+str(hour)+'.pkl'), 'rb'), encoding='bytes')
            route_time_total += route_time
        
        od_route_table = pd.DataFrame({'o_node_id':route_energy_oid,'d_node_id':route_energy_did,'od_time_hour':route_energy_time,'energy':route_energy_total,'dis':route_dis_total,'time':route_time_total})
        od_route_results = od_route_copy.merge(od_route_table,left_on=['od_time_hour','o_node_id','d_node_id'],right_on=['od_time_hour','o_node_id','d_node_id'])

        table = pd.pivot_table(od_route_results, values='energy', index='user_id', aggfunc=np.sum)
        traj_energy = table.to_dict('index')
        pickle.dump(traj_energy, open(os.path.join(self.folder_name,'userTraj_renergy_week.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

        table = pd.pivot_table(od_route_results, values='dis', index='user_id', aggfunc=np.sum)
        traj_dis = table.to_dict('index')
        pickle.dump(traj_dis, open(os.path.join(self.folder_name,'userTraj_rdis_week.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

        table = pd.pivot_table(od_route_results, values='time', index='user_id', aggfunc=np.sum)
        traj_time = table.to_dict('index')
        pickle.dump(traj_time, open(os.path.join(self.folder_name,'userTraj_rtime_week.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    smb = SimMobility()
    smb.calTrajectories()
    smb.calEnergies()
