import SimMobility as smb
import SimAdopter as smapt
import SimBehavior as smbhr
import SimShift as smshf
import itertools
import warnings
warnings.filterwarnings("ignore")

class SimSensitivity():
    def __init__(self, work_l2_access_list,home_l2_access_list,peak_start_list,acceptance_list,max_stay_list):
        super().__init__()
        self.work_l2_access = work_l2_access_list
        self.home_l2_access = home_l2_access_list
        self.peak_start = peak_start_list
        self.acceptance = acceptance_list
        self.max_stay = max_stay_list
    
    def runSimulation(self):
        for i in itertools.product(self.work_l2_access, self.home_l2_access):
            work_l2_access,home_l2_access = i
            # apt = smapt.SimAdopter(work_l2_access,home_l2_access)
            # apt.calUserInformation()
            # apt.calEVDrivers()
            # del apt
            
            # bhr = smbhr.SimBehavior(work_l2_access,home_l2_access)
            # bhr.calGroup()
            # bhr.calBehavior()
            # del bhr

            for j in itertools.product(self.peak_start, self.acceptance, self.max_stay):
                peak_start,acceptance,max_stay = j
                shf = smshf.SimShift(work_l2_access,home_l2_access,peak_start,peak_start+3,acceptance,max_stay)
                shf.calShift()
                shf.calResults()
                del shf

if __name__ == "__main__":
    work_l2_access_list = [0.5]; home_l2_access_list = [0.541]
    peak_start_list = [17]
    acceptance_list = [1]; max_stay_list = [3,5,7,9]
    sms = SimSensitivity(work_l2_access_list,home_l2_access_list,peak_start_list,acceptance_list,max_stay_list)
    sms.runSimulation()
    