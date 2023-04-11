"""Calculate conditional probability P(SPEECh group | Home and/or Work charging access, Annual energy)."""

import pandas as pd
import numpy as np

driver_labels_file_string = '' # from the clustering with n=136
save_folder_location = ''

df = pd.read_csv(driver_labels_file_string, index_col=0)

### P(G)
pg = pd.DataFrame({'pg':np.zeros((136,))})
cts = df['Agglom Cluster Number'].value_counts()
for key, val in cts.items():
    pg.loc[key, 'pg'] = val/len(df)
pg.to_csv(save_folder_location + 'pg_136.csv')


### P(G|Access, Energy)
energy_bins_left = np.arange(0, 5000, 250)
energy_bins_right = np.arange(250, 5001, 250)

pg_HnoWyes = pd.DataFrame(np.zeros((n, 20)), columns=energy_bins_left)
pg_HnoWno = pd.DataFrame(np.zeros((n, 20)), columns=energy_bins_left)
pg_HyesWyes = pd.DataFrame(np.zeros((n, 20)), columns=energy_bins_left)
pg_HyesWno = pd.DataFrame(np.zeros((n, 20)), columns=energy_bins_left)
for i, energy in enumerate(energy_bins_left):
    tmp = df.loc[(df['Total Energy']>=energy)&(df['Total Energy']<energy_bins_right[i])]
    cts = tmp.loc[(tmp['A_flag_work_price']!='0')&(tmp['A_flag_home']=='0')]['Agglom Cluster Number'].value_counts()
    for key, val in cts.items():
        pg_HnoWyes.loc[key, energy] = val/sum(cts)
    cts = tmp.loc[(tmp['A_flag_work_price']!='0')&(tmp['A_flag_home']=='l2')]['Agglom Cluster Number'].value_counts()
    for key, val in cts.items():
        pg_HyesWyes.loc[key, energy] = val/sum(cts)
    cts = tmp.loc[(tmp['A_flag_work_price']=='0')&(tmp['A_flag_home']=='0')]['Agglom Cluster Number'].value_counts()
    for key, val in cts.items():
        pg_HnoWno.loc[key, energy] = val/sum(cts)
    cts = tmp.loc[(tmp['A_flag_work_price']=='0')&(tmp['A_flag_home']=='l2')]['Agglom Cluster Number'].value_counts()
    for key, val in cts.items():
        pg_HyesWno.loc[key, energy] = val/sum(cts)
for data in [pg_HnoWyes, pg_HyesWyes, pg_HnoWno, pg_HyesWno]:
    data[0] = np.copy(data[250]) # fill in first bin because none of the clustered drivers had such low energy
    
pg_HnoWyes.to_csv(save_folder_location + 'pg_HnoWyes_n136.csv')
pg_HyesWyes.to_csv(save_folder_location + 'pg_HyesWyes_n136.csv')
pg_HnoWno.to_csv(save_folder_location + 'pg_HnoWno_n136.csv')
pg_HyesWno.to_csv(save_folder_location + 'pg_HyesWno_n136.csv')