"""Preparing validation files from the speech charging data. 

Note: we only use 'top_zips' aka ZIP codes with at least 36500 sessions or drivers (on average 100 per day throughout the year) to protect privacy, as smaller zip codes with less aggregation of fewer drivers or sessions risks revealing data on individual charging patterns. 
"""

import pandas as pd
import numpy as np

session_data_file_string = ''
driver_labels_file_string = '' # from the clustering with n=136

save_folder_location = ''

df = pd.read_csv(session_data_file_string, index_col=0)
df2 = pd.read_csv(driver_labels_file_string, index_col=0)


# Add column to sessions data with the driver cluster number
df.loc[df.index, 'Agglom Cluster Number 136'] = np.nan
for i in range(136):
    drivers = df2.loc[df2['Agglom Cluster Number']==i]['Unique Driver ID'].values
    inds = df.loc[df['Driver ID'].isin(drivers)].index
    df.loc[inds, 'Agglom Cluster Number 136'] = i
df = df[df['Driver ID'].isin(df2['Unique Driver ID'].values)]

subset = df.loc[:, ['Driver ID', 'Driver Zip', 'Agglom Cluster Number 136']].copy(deep=True)

### Output 1: P(G|zip)
pzip = pd.DataFrame({'Zip':subset['Driver Zip'].unique(), 'Weight':np.zeros((len(subset['Driver Zip'].unique()),))})
cts = subset['Driver Zip'].value_counts() / len(subset)
for key, val in cts.items():
    idx = pzip[pzip['Zip']==key].index
    pzip.loc[idx, 'Weight'] = val
pzip.to_csv(save_folder_location + 'pzip.csv')

### Output 2: P(G) in Top ZIPs for validation
top_zips = df['Zip Code'].value_counts()[df['Zip Code'].value_counts()>36500].keys()
top_driver_zips = df['Driver Zip'].value_counts()[df['Driver Zip'].value_counts()>36500].keys()

pg_zip_driver = pd.DataFrame({'G':np.arange(0, 136)})
for zipcode in top_driver_zips:
    pg_zip_driver[zipcode] = 0
    subset_here = subset[subset['Driver Zip']==zipcode]
    cts = subset_here['Agglom Cluster Number 136'].value_counts()
    for key, val in cts.items():
        pg_zip_driver.loc[key, zipcode] = val/len(subset_here)

pg_zip_sessions = pd.DataFrame({'G':np.arange(0, 136)})
for zipcode in top_zips:
    pg_zip_sessions[zipcode] = 0
    subset_here = df[df['Zip Code']==zipcode]
    cts = subset_here['Agglom Cluster Number 136'].value_counts()
    for key, val in cts.items():
        pg_zip_sessions.loc[key, zipcode] = val/len(subset_here)

pg_zip_driver.to_csv(save_folder_location + 'validation136_pg_zip_topzips_bydriverhome.csv')
pg_zip_sessions.to_csv(save_folder_location + 'validation136_pg_zip_topzips_bysessionlocation.csv')

### Functions to calculate load from data point (start time, energy delivered, power)

def raw_data_load(df_subset, num_ts=1440):
    
    all_loads = {}

    start_times = (1/60)*df_subset['start_seconds'].values.astype(int)
    energies = df_subset['Energy (kWh)'].values
    rates = df_subset['Max Power'].values

    end_times, load = end_times_and_load_mixedrates(start_times, energies, rates, 60, 1440)
    all_loads['total'] = load
    
    wp_set = df_subset[df_subset['POI Category']=='Workplace']
    res_set = df_subset[df_subset['POI Category']=='Single family residential']
    mud_set = df_subset[df_subset['POI Category']=='Mul tifamily Home Service']
    other_set = df_subset[df_subset['POI Category'].isin(['Education', 'Utility', 'Retail', 'Parking', 'Healthcare', 'Municipal', 'Multifamily Commercial', 'Parks and Recreation', 'Hospitality', 'Government (Fed, State)'])]
    other_slow_set = other_set[other_set['Max Power']<20]
    other_fast_set = other_set[other_set['Max Power']>=20]
    labels = ['Residential L2', 'Workplace L2', 'MUD L2', 'Public L2', 'Public DCFC']
    
    for i, data in enumerate([res_set, wp_set, mud_set, other_slow_set, other_fast_set]):
        start_times = (1/60)*data['start_seconds'].values.astype(int)
        energies = data['Energy (kWh)'].values
        rates = data['Max Power'].values

        end_times, load = end_times_and_load_mixedrates(start_times, energies, rates, 60, 1440)
        all_loads[labels[i]] = load
        
    return all_loads

def end_times_and_load_mixedrates(start_times, energies, rate, time_steps_per_hour, num_time_steps):

    num = np.shape(start_times)[0]
    load = np.zeros((num_time_steps,))
    end_times = np.zeros(np.shape(start_times)).astype(int)

    lengths = (time_steps_per_hour * energies / rate).astype(int)
    extra_charges = energies - lengths * rate / time_steps_per_hour
    inds1 = np.where((start_times + lengths) > num_time_steps)[0]
    inds2 = np.delete(np.arange(0, np.shape(end_times)[0]), inds1)

    end_times[inds1] = (np.minimum(start_times[inds1].astype(int)+lengths[inds1]-num_time_steps, num_time_steps)).astype(int)
    end_times[inds2] = (start_times[inds2] + lengths[inds2]).astype(int)
    inds3 = np.where(end_times >= num_time_steps)[0]
    inds4 = np.delete(np.arange(0, np.shape(end_times)[0]), inds3)

    for i in range(len(inds1)):
        idx = int(inds1[i])
        load[np.arange(int(start_times[idx]), num_time_steps)] += rate[idx] * np.ones((num_time_steps - int(start_times[idx]),))
        load[np.arange(0, end_times[idx])] += rate[idx] * np.ones((end_times[idx],))
    for i in range(len(inds2)):
        idx = int(inds2[i])
        load[np.arange(int(start_times[idx]), end_times[idx])] += rate[idx] * np.ones((lengths[idx],))
    load[0] += np.sum(extra_charges[inds3] * time_steps_per_hour)
    for i in range(len(inds4)):
        load[end_times[int(inds4[i])]] += extra_charges[int(inds4[i])] * time_steps_per_hour

    return end_times, load

### Output 3: Full and Normalized ZIP-level demand, grouped by session location

days = df['start_day'].unique() 
top_zips = df['Zip Code'].value_counts()[df['Zip Code'].value_counts()>36500].keys()
top_zip_dfs = {'Counts':{}, 'Total':{}, 'Norm1':{}, 'Norm2':{}}
for j in range(len(top_zips)):
    print('Zip: ', top_zips[j], '. Number of sessions: ', df['Zip Code'].value_counts()[top_zips[j]])
    top_zip_dfs['Counts'][j] = pd.DataFrame(np.zeros((len(days), 2)), columns=['Num Sessions', 'Num Drivers'])
    top_zip_dfs['Total'][j] = np.zeros((1440, 6, len(days)))
    top_zip_dfs['Norm1'][j] = np.zeros((1440, 6, len(days)))
    top_zip_dfs['Norm2'][j] = np.zeros((1440, 6, len(days)))
    for i, day in enumerate(days):
        subset = df.loc[(df['Zip Code']==top_zips[j])&(df['start_day']==day)]
        if len(subset) > 0:
            loads = raw_data_load(subset)
            top_zip_dfs['Total'][j][:, :, i] = pd.DataFrame(loads).values
            top_zip_dfs['Counts'][j].loc[i, 'Num Sessions'] = len(subset)
            top_zip_dfs['Counts'][j].loc[i, 'Num Drivers'] = len(subset['Driver ID'].unique())
            top_zip_dfs['Norm1'][j][:, :, i] = pd.DataFrame(loads).values / top_zip_dfs['Counts'][j].loc[i, 'Num Sessions']
            top_zip_dfs['Norm2'][j][:, :, i] = pd.DataFrame(loads).values / top_zip_dfs['Counts'][j].loc[i, 'Num Drivers']
            
here_days = top_zip_dfs['Counts'][0][top_zip_dfs['Counts'][0]['Num Sessions']!=0].index
for i, name in enumerate(['total', 'sfh_l2', 'work_l2', 'mud_l2', 'pub_l2', 'pub_l3']):
    zip_loads = pd.DataFrame(np.zeros((1440, len(top_zips))), columns=top_zips)
    for j, zipcode in enumerate(top_zips):
        zip_loads[zipcode] = top_zip_dfs['Norm1'][j][:, :, here_days].mean(axis=2)[:, i]
    zip_loads.to_csv(save_folder_location + 'normalized_load_by_zipcode_'+name+'_topzips_sessionlocation.csv')

#### Calculate statistics on mean sessions per zip
zip_stats = pd.DataFrame({'Zip Code': top_zips, 'Mean Daily Sessions Here': np.zeros((len(top_zips),)), 
                          'Mean Daily Drivers Here':np.zeros((len(top_zips),))})
for i in zip_stats.index:
    zip_stats.loc[i, 'Mean Daily Sessions Here'] = top_zip_dfs['Counts'][i]['Num Sessions'].mean()
    zip_stats.loc[i, 'Mean Daily Drivers Here'] = top_zip_dfs['Counts'][i]['Num Drivers'].mean()
zip_stats.to_csv(save_folder_location + 'zipcode_stats_for_validation_topzips_sessionlocation.csv')


### Output 4: Full and Normalized ZIP-level demand, grouped by driver home location

days = df['start_day'].unique() 
top_zips = df['Driver Zip'].value_counts()[df['Driver Zip'].value_counts()>36500].keys()
top_zip_dfs = {'Counts':{}, 'Total':{}, 'Norm1':{}, 'Norm2':{}}
for j in range(len(top_zips)):
    num_drivers_home = len(df[df['Driver Zip']==top_zips[j]]['Driver ID'].unique())
    print('Zip: ', top_zips[j], '. Number of drivers: ', num_drivers_home)
    top_zip_dfs['Counts'][j] = pd.DataFrame(np.zeros((len(days), 2)), columns=['Num Sessions', 'Num Drivers'])
    top_zip_dfs['Total'][j] = np.zeros((1440, 6, len(days)))
    top_zip_dfs['Norm2'][j] = np.zeros((1440, 6, len(days)))
    for i, day in enumerate(days):
        subset = df.loc[(df['Driver Zip']==top_zips[j])&(df['start_day']==day)]
        if len(subset) > 0:
            loads = raw_data_load(subset)
            top_zip_dfs['Total'][j][:, :, i] = pd.DataFrame(loads).values
            top_zip_dfs['Counts'][j].loc[i, 'Num Sessions'] = len(subset)
            top_zip_dfs['Counts'][j].loc[i, 'Num Drivers'] = num_drivers_home
            top_zip_dfs['Norm2'][j][:, :, i] = pd.DataFrame(loads).values / top_zip_dfs['Counts'][j].loc[i, 'Num Drivers']

here_days = top_zip_dfs['Counts'][0][top_zip_dfs['Counts'][0]['Num Sessions']!=0].index
for i, name in enumerate(['total', 'sfh_l2', 'work_l2', 'mud_l2', 'pub_l2', 'pub_l3']):
    zip_loads = pd.DataFrame(np.zeros((1440, len(top_zips))), columns=top_zips)
    for j, zipcode in enumerate(top_zips):
        zip_loads[zipcode] = top_zip_dfs['Norm2'][j][:, :, here_days].mean(axis=2)[:, i]
    zip_loads.to_csv(save_folder_location + 'normalized_load_by_zipcode_'+name+'_topzips_driverhome.csv')
        
#### Calculate statistics on mean sessions per zip
zip_stats = pd.DataFrame({'Zip Code': top_zips, 'Mean Daily Sessions Here': np.zeros((len(top_zips),)), 
                          'Mean Daily Drivers Here':np.zeros((len(top_zips),))})
for i in zip_stats.index:
    zip_stats.loc[i, 'Mean Daily Sessions Here'] = top_zip_dfs['Counts'][i]['Num Sessions'].mean()
    zip_stats.loc[i, 'Mean Daily Drivers Here'] = top_zip_dfs['Counts'][i]['Num Drivers'].mean()
zip_stats.to_csv(save_folder_location + 'zipcode_stats_for_validation_topzips_driverhome.csv')

