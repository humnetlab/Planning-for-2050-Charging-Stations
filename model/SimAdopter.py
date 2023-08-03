import random
import pickle
import numpy as np
import pandas as pd
import os
import numpy as np
import random as rand
import warnings
warnings.filterwarnings("ignore")

class SimAdopter():
    def __init__(self, work_l2_access,home_l2_access):
        super().__init__()
        self.work_l2_access = work_l2_access
        self.home_l2_access = home_l2_access
        self.mobility_folder_name = os.path.join('..','result','mobility')
        self.adopter_folder_name = os.path.join('..','result','adopter','work_'+str(self.work_l2_access)+'home_'+str(self.home_l2_access))
        if not os.path.exists(self.adopter_folder_name):
            os.makedirs(self.adopter_folder_name)
    
    def collectCensusTract(self):
        housetype = pd.read_csv(os.path.join('..','data','census','ACS1101/ACSST5Y2019.S1101_data_with_overlays_2022-04-05T180436.csv'),skiprows=[1])
        income = pd.read_csv(os.path.join('..','data','census','ACS1901/ACSST5Y2019.S1901_data_with_overlays_2022-04-05T181030.csv'),skiprows=[1])
        housetype = housetype[['GEO_ID','S1101_C01_014E','S1101_C01_015E','S1101_C01_016E']]
        income = income[['GEO_ID','S1901_C01_013E']]
        census = pd.merge(left=income, right=housetype, left_on = 'GEO_ID', right_on='GEO_ID')
        census.columns = ['GEO_ID','INCOME_M','SFH_M','MFH_M','MOT_M']
        census.to_csv(os.path.join('..','data','census','census_ac.csv'),index=False)

    def findUserHomeTract(self):
        # load the point to tract 
        pointsInTracts = pickle.load(open(os.path.join('..','data','census','pointsInTracts.pkl'), 'rb'))

        # load the simulated data
        timegeo_week_path = os.path.join('..','data','timegeo_week','')
        simFiles = [timegeo_week_path + f for f in os.listdir(timegeo_week_path) if os.path.isfile(os.path.join(timegeo_week_path, f)) and 'txt' in f]
        simFiles = sorted(simFiles)

        userHomeTract = {}
        # we only consider commuters here
        userCount = 0
        for f in simFiles:
            data = open(f, 'r')
            for line in data:
                line = line.strip().split(' ')
                # print line
                if len(line) == 1:
                    userCount += 1
                    perID = int(line[0].split('-')[0])
                    findHome = 0
                else:
                    # extract lon, lat
                    lon = float(line[2])
                    lat = float(line[3])
                    stayLabel = line[1]
                    if stayLabel == 'h' and findHome==0:
                        try:
                            userHomeTract[perID] = pointsInTracts[(lon, lat)]
                            findHome = 1
                        except:
                            continue
                    else:
                        continue

        print("# of users without home : %d / %d" % (len(userHomeTract), userCount))
        pickle.dump(userHomeTract, open(os.path.join(self.adopter_folder_name,'userHomeTract.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    def userInfCollect(self):
        userTraj_label = pickle.load(open(os.path.join(self.mobility_folder_name,'userTraj_label_week.pkl'), 'rb'))
        userTraj_zipcode = pickle.load(open(os.path.join(self.mobility_folder_name,'userTraj_zipcode_week.pkl'), 'rb'))
        user_inf = {}
        count_home = 0; count_work = 0
        for user in userTraj_label:
            zipcodes = userTraj_zipcode[user]
            staylabels = userTraj_label[user]
            homeZipcode = ''
            workZipcode = ''
            for i in range(len(staylabels)):
                stayLabel = staylabels[i]
                zipcode = zipcodes[i]
                if stayLabel[0] == 'h' and homeZipcode == '' and zipcode != '':
                    homeZipcode = zipcode
                    count_home +=1 
                if stayLabel[0] == 'w' and workZipcode == '' and zipcode != '':
                    workZipcode = zipcode
                    count_work +=1 
            user_inf[user] = [homeZipcode, workZipcode]
            
        print("The user with zipcode level information:",len(user_inf))
        print("The user with home information:",count_home)
        print("The user with work information:",count_work)
        
        pickle.dump(user_inf, open(os.path.join(self.adopter_folder_name,'userInfor.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    def individualDistance(self):
        userHomeTract = pickle.load(open(os.path.join(self.adopter_folder_name,'userHomeTract.pkl'), 'rb'))
        user_dis = pickle.load(open(os.path.join(self.mobility_folder_name,'userTraj_rdis_week.pkl'), 'rb'), encoding='bytes')

        count_0 = 0; count_100 = 0
        userDistances = {}
        for user in userHomeTract:
            try:
                distance = user_dis[user]['dis']/7  #miles
            except:
                distance = 0
            if distance == 0:
                count_0 = count_0 + 1
                continue
            if distance > 100:
                count_100 = count_100 + 1
                continue  # we remove abnormal long trips
            userDistances[user] = distance

        # histogram of income
        distances = list(userDistances.items())
        bins = [0, 15, 30, 45, 100]
        hist = np.histogram(distances, bins)
        distri_distance = np.divide(hist[0], float(len(distances)))

        # calculate probability of individual income falls in one bin
        for user in userDistances:
            distance = min(userDistances[user], 99.9)
            distance = max(distance, 0)
            bin = int(np.digitize(distance, bins))
            p_distance = distri_distance[bin-1]
            userDistances[user] = [distance, p_distance]

        print("# of users with distance : ", len(userDistances))
        print("# of users with distance 0 : ", count_0)
        print("# of users with distance 100 : ", count_100)
        pickle.dump(userDistances, open(os.path.join(self.adopter_folder_name,'userDistances.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)


    def individualConsumption(self):
        userHomeTract = pickle.load(open(os.path.join(self.adopter_folder_name,'userHomeTract.pkl'), 'rb'))
        user_energy = pickle.load(open(os.path.join(self.mobility_folder_name,'userTraj_renergy_week.pkl'), 'rb'), encoding='bytes')

        count_0 = 0; count_30 = 0
        userConsumptions = {}
        for user in userHomeTract:
            try:
                consumption = user_energy[user]['energy']/7  #km
            except:
                consumption = 0
            if consumption == 0:
                count_0 = count_0+1
                continue
            if consumption > 30:
                count_30 = count_30+1
                continue

            userConsumptions[user] = consumption

        # histogram of income
        consumption = list(userConsumptions.items())
        bins = [0, 5, 10, 15, 20, 25, 30]
        hist = np.histogram(consumption, bins)
        distri_consumption = np.divide(hist[0], float(len(consumption)))

        # calculate probability of individual income falls in one bin
        for user in userConsumptions:
            consumption = min(userConsumptions[user], 30-0.1)
            consumption = max(consumption, 0)
            bin = int(np.digitize(consumption, bins))
            p_consumption = distri_consumption[bin-1]
            userConsumptions[user] = [consumption, p_consumption]

        print("# of users with consumption : ", len(userConsumptions))
        print("# of users with consumption 0 : ", count_0)
        print("# of users with consumption 30 : ", count_30)
        # save individual income information
        pickle.dump(userConsumptions, open(os.path.join(self.adopter_folder_name,'userConsumptions.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    def get_tract_income(self,meanIncome):
        mean = float(meanIncome)
        std = meanIncome/4
        pick = int(rand.normalvariate(mean,std))
        pick = max(mean - std, pick)
        pick = min(mean + std, pick)
        return int(pick)

    def individualIncome(self):
        userHomeTract = pickle.load(open(os.path.join(self.adopter_folder_name,'userHomeTract.pkl'), 'rb'))
        tractsInfor = np.genfromtxt(os.path.join('..','data','census','sfbay_tract_infor.csv'), delimiter=',', skip_header=1, dtype=None)

        tractsIncome = {}
        for t in tractsInfor:
            geoID = str(int(t[1])).zfill(11)
            vur = float(t[-3])
            income = float(t[-1])
            tractsIncome[geoID] = income

        # for each user, we find the home tract first, and then assign income information
        count = 0
        userIncomes = {}
        for user in userHomeTract:  # [stayLabel, lon, lat]
            homeTract = userHomeTract[user]
            try:
                meanIncome = tractsIncome[homeTract]
            except:
                count = count+1
                continue
            personIncome = self.get_tract_income(meanIncome)
            userIncomes[user] = personIncome

        # histogram of income
        incomes = list(userIncomes.items())
        bins = [0, 50000, 100000, 150000, 250000]
        hist = np.histogram(incomes, bins)
        distri_income = np.divide(hist[0], float(len(incomes)))

        # calculate probability of individual income falls in one bin
        for user in userIncomes:
            income = min(userIncomes[user], 250000-0.01)
            income = max(income, 0)
            bin = int(np.digitize(income, bins))
            p_income = distri_income[bin-1]
            userIncomes[user] = [income, p_income]
        
        print("# of users with income : ", len(userIncomes))
        print("# of users without income : ", count)
        # save individual income information
        pickle.dump(userIncomes, open(os.path.join(self.adopter_folder_name,'userIncomes.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)


    def individualEV(self):
        userHomeTract = pickle.load(open(os.path.join(self.adopter_folder_name,'userHomeTract.pkl'), 'rb'))
        tractsInfor = np.genfromtxt(os.path.join('..','data','census','sfbay_tract_infor.csv'), delimiter=',', skip_header=1, dtype=None)
        tractsVUR = {}
        for t in tractsInfor:
            geoID = str(int(t[1])).zfill(11)
            vur = float(t[-3])
            tractsVUR[geoID] = vur

        userVehicleInfor = {}
        for user in userHomeTract:  # [stayLabel, lon, lat]
            homeTract = userHomeTract[user]
            userVehicleInfor[user] = tractsVUR[homeTract]*0.0062
        
        print("# of users with vehicle : ", len(userVehicleInfor))
        pickle.dump(userVehicleInfor, open(os.path.join(self.adopter_folder_name,'userVehicles.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    def get_tract_house(self,meanHouse):
        mean = float(meanHouse)
        pick = np.random.choice(2, 1, p=[mean, 1-mean])
        return int(pick)

    def individualHouse(self):
        userHomeTract = pickle.load(open(os.path.join(self.adopter_folder_name,'userHomeTract.pkl'), 'rb'))
        tractsInfor = pd.read_csv(os.path.join('..','data','census','census_ac.csv'))
        tractsHouse = {}
        for t in range(len(tractsInfor)):
            geoID = tractsInfor['GEO_ID'].iloc[t][-11:]
            try:
                house = float(tractsInfor['SFH_M'].iloc[t])/(float(tractsInfor['SFH_M'].iloc[t])+float(tractsInfor['MFH_M'].iloc[t])+1e-5)
                tractsHouse[geoID] = house
            except:
                continue
        # for each user, we find the home tract first, and then assign income information
        count = 0; tract_nan = []
        userHouses = {}
        for user in userHomeTract:  # [stayLabel, lon, lat]
            homeTract = userHomeTract[user]
            try:
                meanHouse = float(tractsHouse[homeTract])
                personHouse = int(self.get_tract_house(meanHouse))
                userHouses[user] = personHouse
            except:
                count = count+1
                tract_nan.append(homeTract)
                continue

        # histogram of income
        houses = list(userHouses.items())
        bins = [0, 0.9, 1.1]
        hist = np.histogram(houses, bins)
        distri_house = np.divide(hist[0], float(len(houses)))

        # calculate probability of individual income falls in one bin
        for user in userHouses:
            house = userHouses[user]
            bin = int(np.digitize(house, bins))
            p_house = distri_house[bin-1]
            userHouses[user] = [house, p_house]

        print("# of users with house type : ", len(userHouses))
        print("# of users without house type : ", count)
        pickle.dump(userHouses, open(os.path.join(self.adopter_folder_name,'userHouses.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    def individualStay(self):
        userHomeTract = pickle.load(open(os.path.join(self.adopter_folder_name,'userHomeTract.pkl'), 'rb'))
        user_stay = pickle.load(open(os.path.join(self.mobility_folder_name,'userTraj_label_week.pkl'), 'rb'), encoding='bytes')

        count_0 = 0; count_100 = 0
        userStays = {}
        for user in user_stay:
            try:
                stay = len(user_stay[user]) #miles
            except:
                stay = 0
            if (stay <= 14) and (stay>0):
                count_100 = count_100 + 1
                continue
            elif (stay==0):
                count_0 = count_0 + 1
                continue
            userStays[user] = 0

        print("# of users with stay : ", len(userStays))
        print("# of users with stay 0 : ", count_0)
        print("# of users with stay < 7 : ", count_100)
        pickle.dump(userStays, open(os.path.join(self.adopter_folder_name,'userStays.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

    # def get_access_p_income_house(self,income,house):
    #     if income<60000 and house == 0:
    #         p = 0.40
    #     elif income<60000 and house == 1:
    #         p = 0.22
    #     elif income>60000 and income<100000 and house == 0:
    #         p = 0.46
    #     elif income>60000 and income<100000 and house == 1:
    #         p = 0.28
    #     elif income>100000 and house == 0:
    #         p = 0.50
    #     else:
    #         p = 0.34
    #     return p

    def get_access_p_income_house(self,income,house):
        if income<60000 and house == 0:
            p = 0.72
        elif income<60000 and house == 1:
            p = 0.29
        elif income>60000 and income<100000 and house == 0:
            p = 0.78
        elif income>60000 and income<100000 and house == 1:
            p = 0.36
        elif income>100000 and house == 0:
            p = 0.85
        else:
            p = 0.43
        return p
    

    def get_tract_acess(self,meanAccess,workAccess):
        # home charging access
        mean_l1 = float(meanAccess)
        mean_l2 =  self.home_l2_access#np.clip(0.54*mean_l1,0,1)
        pick_home_l1 = np.random.choice(2, 1, p=[1-mean_l1, mean_l1])[0]
        pick_home_l2 = 0
        if pick_home_l1 == 1:
            pick_home_l2 = np.random.choice(2, 1, p=[1-mean_l2, mean_l2])[0]
        if pick_home_l2 == 1:
            pick_home_l1 = 0
        if pick_home_l1 or pick_home_l2:
            pick_home = 1
        else:
            pick_home = 0
        # workplace charging access
        if len(workAccess)<5:
            pick_work = 0
        else:
            pick_work = np.random.choice(2, 1, p=[1-self.work_l2_access,self.work_l2_access])[0]

        if pick_home == 0 and pick_work == 0:
            pick = 0
        elif pick_home == 1 and pick_work == 0:
            pick = 1
        elif pick_home == 0 and pick_work == 1:
            pick = 2
        else:
            pick = 3
        return int(pick),pick_home_l1,pick_home_l2,pick_work

    def get_ev_p_distance(self,dist):
        bins = 15
        if dist<bins:
            p = 0.14
        elif dist<2*bins:
            p = 0.50
        elif dist<3*bins:
            p = 0.28
        elif dist<7*bins:
            p = 0.08
        else:
            p = 0
        return p

    def get_ev_p_income(self,income):
        unknown = 0.18
        if income<50000:
            p = 0.02
        elif income<100000:
            p = 0.13
        elif income<150000:
            p = 0.20
        elif income<250000:
            p = 0.47
        else:
            p = 0
        return p / (1-unknown)

    def userInfComplete(self):
        userInfor = pickle.load(open(os.path.join(self.adopter_folder_name,'userInfor.pkl'), 'rb'))
        userHouseInfor = pickle.load(open(os.path.join(self.adopter_folder_name,'userHouses.pkl'), 'rb'))
        userIncomeInfor = pickle.load(open(os.path.join(self.adopter_folder_name,'userIncomes.pkl'), 'rb'))
        userConsumptionInfor = pickle.load(open(os.path.join(self.adopter_folder_name,'userConsumptions.pkl'), 'rb'))
        userDistanceInfor = pickle.load(open(os.path.join(self.adopter_folder_name,'userDistances.pkl'), 'rb'))
        userHomeTract = pickle.load(open(os.path.join(self.adopter_folder_name,'userHomeTract.pkl'), 'rb'))
        userVehicleInfor = pickle.load(open(os.path.join(self.adopter_folder_name,'userVehicles.pkl'), 'rb'))
        users = userHomeTract.keys()
        
        count = 0; user_nan = []
        final_user_inf = []
        for user in users:
            try:
                userInf = userInfor[user]
                incomeInf = userIncomeInfor[user]
                consumptionInf = userConsumptionInfor[user]
                houseInf = userHouseInfor[user]
                distanceInf = userDistanceInfor[user]
                tractInf = userHomeTract[user]
                vehicleInf = userVehicleInfor[user]

                income2019 = 1.6*incomeInf[0]
                personAccess,pick_home_l1,pick_home_l2,pick_work = self.get_tract_acess(self.get_access_p_income_house(income2019,houseInf[0]),userInf[1])
                p_distance_ev = self.get_ev_p_distance(distanceInf[0])
                p_income_ev = self.get_ev_p_income(incomeInf[0])
                p_distance = distanceInf[1]
                p_income = incomeInf[1]
                p_ev = vehicleInf 
                ev_prob = p_income_ev*p_distance_ev*p_ev/(p_distance*p_income)

                row = [user]+ userInf+ [tractInf, income2019, houseInf[0] , consumptionInf[0] , distanceInf[0], personAccess, pick_home_l1, pick_home_l2, pick_work, ev_prob]
                final_user_inf.append(row)
            except:
                user_nan.append(user)
                count = count + 1
                continue


        final_user_inf = pd.DataFrame(final_user_inf, columns = ['userID', 'homeZipcode', 'workZipcode', 'homeTract', 'personIncome',
                                                            'personHouse','personConsumption', 'personDistance', 'personAccess', 'personAccess_hl1','personAccess_hl2','personAccess_w','ev_prob'])
        print("# of users with info : ", len(final_user_inf))
        print("# of users without info : ", count)
        pickle.dump(final_user_inf, open(os.path.join(self.adopter_folder_name,'df_user_inf.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
    
    def calUserInformation(self):
        self.collectCensusTract()
        self.findUserHomeTract()
        self.userInfCollect()
        self.individualDistance()
        self.individualConsumption()
        self.individualIncome()
        self.individualHouse()
        self.individualEV()
        self.userInfComplete()
    
    def getEVprobabilities(self):
        df_user_inf = pickle.load(open(os.path.join(self.adopter_folder_name,'df_user_inf.pkl'), 'rb'), encoding='bytes')
        df_user_inf = df_user_inf.values.tolist()
        totalUsersInBay = len(df_user_inf)
        print("Total users in SFBay with information: ", totalUsersInBay)
        EVflows = {}
        allUsersEVprob = []
        totalEVs = 0
        for row in df_user_inf:
            userID = int(row[0])
            if row[1]=='':
                continue
            homeZipcode = int(row[1])
            ev_prob = float(row[-1])
            totalEVs += ev_prob
            allUsersEVprob.append([userID,ev_prob])
            if homeZipcode not in EVflows:
                EVflows[homeZipcode] = [[userID, ev_prob]]
            else:
                EVflows[homeZipcode].append([userID, ev_prob])

        totalEVs = int(totalEVs)
        print("Total # EV drivers in 2013: ", totalEVs)
        return allUsersEVprob, totalEVs
        
    def selectEVDrivers(self, desiredEVs, yearName):
        allUsersEVprob, totalEVs = self.getEVprobabilities()
        if desiredEVs < 0:
            scaleFactor = 1
        else:
            scaleFactor = desiredEVs/totalEVs
        selecEVs = []
        totalEV_after = 0
        for row in allUsersEVprob:
            userid, ev_prob = row
            ev_prob = ev_prob*scaleFactor
            randnum = random.uniform(0, 1)
            if ev_prob >= randnum:
                selecEVs.append(userid)
        totalEV_after = len(selecEVs)
        print("# of EVs : %.2f" % desiredEVs)
        print("Total # EV drivers selected: ", totalEV_after)
        if desiredEVs < 0:
            saveFileName = os.path.join(self.adopter_folder_name,'selected_EV_Drivers_raw.pkl')
        else:
            saveFileName = os.path.join(self.adopter_folder_name,'selected_EV_Drivers_' + str(yearName) + 'p.pkl')
        pickle.dump(selecEVs, open(saveFileName, 'wb'), pickle.HIGHEST_PROTOCOL)

    def calEVDrivers(self):
        self.selectEVDrivers(1500000*1.59, 0.3)
        self.selectEVDrivers(5000000*5.2, 0.5)
        self.selectEVDrivers(186355*0.69281241, 0)
        self.selectEVDrivers(186355, 1)
        self.selectEVDrivers(1171033*1.44, 2)
        self.selectEVDrivers(2342066*2, 3)
        self.selectEVDrivers(3513099*2.7, 4)
        self.selectEVDrivers(4684132*4.35, 5)
        self.selectEVDrivers(5855165*100, 6)

if __name__ == "__main__":
    smapt = SimAdopter(0.50,0.54)
    smapt.calUserInformation()
    smapt.calEVDrivers()
