import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib 


# plt.style.use('seaborn')
class sci_no_act:
    '''
    Generating synthetic data for the SCI individual
    Time granularity: the minimum time unit used is tu, which means we
    will take the average within each tu for each signal

    '''

    def __init__(self, tu, n_hrs, tau_max):  # age, gender, weight, loi, ht, tsi, smoke

        '''
        demographic features (background factors)
        '''

        # todo to make the basline dependent of age gender...
        self.tu = tu  # the minimum time unit in mins
        self.n_hrs = n_hrs  # number of hours recorded
        self.tau_max = tau_max  # maximum time lag
        self.tlength = self.n_hrs * 60 * self.tu  # total time span of the observation
        # self.t_start = 0  # starting time of recording
        # self.age = age  # age of the patient in years
        # self.gender = gender  # gender of the patient
        # self.weight = weight  # weight of the patient
        # self.loi = loi  # level of injuries 1-30
        # self.ht = ht  # hypertensive episode
        # self.tsi = tsi  # time since injury in years
        # self.smoke = smoke  # smoking habit 0/1

        '''
        dynamic signals 
        '''

        '''
        activities: 0: transfer; 1: 
        '''
        # self.act = np.zeros(self.tlength + self.tau_max)
        # activities recorded of the sci patient encoded as 0-9 from light to heavy
        # 0-sleep; 1-assisted propulsion; 2-using computer; 3-using phone; 4-talking;
        # 5-washing; 6-pressure relief; 7-transfer; 8-self propulsion; 9-exercise
        self.bld = np.zeros(self.tlength + self.tau_max)  # stimulus from bladder distension: 0/1
        self.ad = np.zeros(self.tlength + self.tau_max)  # onsets of autonomic dysreflexia: 0/1
        self.sbp_base = np.random.uniform(90, 110)
        self.sbp = np.full(self.tlength + self.tau_max, self.sbp_base)  # systolic blood pressure
        self.dbp_base = np.random.uniform(50, 60)
        self.dbp = np.full(self.tlength + self.tau_max, self.dbp_base)  # diastolic blood pressure
        self.rr_base = np.random.uniform(12, 20)
        self.rr = np.full(self.tlength + self.tau_max, self.rr_base)  # respiration rate
        self.hr_base = np.random.uniform(50, 80)  # heart rate baseline
        self.hr = np.full(self.tlength + self.tau_max, self.hr_base)  # heart rate
        self.bt_base = np.random.uniform(35.6, 36.0)  # body temperature baseline
        self.bt = np.full(self.tlength + self.tau_max, self.bt_base)  # body temperature: Celsius degrees
        self.spo_base = np.random.uniform(85, 90)
        self.spo = np.full(self.tlength + self.tau_max, self.spo_base)  # blood oxygen levels .%
        self.eda_base = np.random.uniform(0.3, 0.7)
        self.eda = np.full(self.tlength + self.tau_max, self.eda_base)  # records of sweating: 0/1

    def get_signals(self, seed):

        np.random.seed(seed)

        # initialization
        # number of activities and time to switch
        # nr_act = 6  # np.random.randint(6,12)
        # print(nr_act)
        # switch_pt = np.random.randint(0, self.tlength + self.tau_max, nr_act)
        # switch_pt.sort()
        # # print(switch_pt)
        # self.act[:switch_pt[0]] = 5  # can be hard-fixed to transport --> wash --> ...
        # self.act[switch_pt[-1]:] = 9  # np.random.randint(0,10) #can be hard-fixed to wash --> transport --> sleep
        # cur_act = None
        # for i in range(nr_act - 1):
        #     temp_act = np.random.randint(0, 10)
        #     if temp_act == cur_act:
        #         temp_act = np.random.randint(0, 10)
        #     self.act[switch_pt[i]:switch_pt[i + 1]] = temp_act
            # bladder distension
        self.bld_filled = random.uniform(250, 420)
#         bld_period = 10 * self.tu  # bladder extension last for 60 mins, number of bladder distension is 1 time per recording
        for i in range(self.tlength + self.tau_max):
            self.bld[i] = self.bld_filled/(self.tlength + self.tau_max)*(i-self.tau_max)
        # define the criterion of an onset of AD
        self.diff = random.uniform(20, 40)
        seed = 1
        # noise = np.random.normal(0, 0.1)
        for t in range(self.tau_max, self.tlength + self.tau_max):
            self.bt[t] += ar(self.bt, t, 6)/self.bt[0] - (self.eda[t - 3]-self.eda_base) * 0.01 + np.random.normal(0, 0.01)
            # ad this is a redundant node as AD is now defined by sbp
            self.ad[t] = indic(self.sbp[t - 1] - self.sbp_base, self.diff)
            # systolic blood pressure
            scale_sbp = random.uniform(0.2, 0.4)
            self.sbp[t] += ar(self.sbp, t, 6) / self.sbp[0] + self.bld[t - 4] * scale_sbp + np.random.normal(0, 1)
            # diastolic blood pressure
            scale_dbp = random.uniform(0.2, 0.4)
            self.dbp[t] += ar(self.dbp, t, 6) / self.dbp[0] + self.bld[t - 4] * scale_dbp + np.random.normal(0, 1)
            # sweating (eda)
            scale_eda = np.random.uniform(0,0.0005)
            self.eda[t] += ar(self.eda, t, 4)/ 4 + (self.bt[t] - self.bt_base)/10 + (self.sbp[t - 1] - self.sbp_base)* scale_eda + np.random.normal(0, 1)
            # heart rate
            scale_hr = random.uniform(0.5, 1)
            self.hr[t] += ar(self.hr, t, 6) / self.hr[0] + max(0, (self.sbp[t] - self.sbp_base)) * scale_hr + np.random.normal(0, 1)
            # respiration rate
            scale_rr = random.uniform(0.2, 0.25)
            self.rr[t] += ar(self.rr, t, 4) / self.rr[0] + (self.hr[t - 1] - self.hr_base) * scale_rr + np.random.normal(0, 0.01)
            # blood oxygen level
            scale_spo = random.uniform(0.15, 0.25)
            self.spo[t] += ar(self.spo, t, 4) / self.spo[0] + self.rr[t - 2] * scale_spo + np.random.normal(0, 0.01)

        return {
            "sbp": self.sbp[self.tau_max:, ],
            "dbp": self.dbp[self.tau_max:, ],
            "heart_rate": self.hr[self.tau_max:, ],
            "resp_rate": self.rr[self.tau_max:, ],
            "temp_skin": self.bt[self.tau_max:, ],
            "spo2": self.spo[self.tau_max:, ],
            "bld_vol": self.bld[self.tau_max:, ],
            "eda": self.eda[self.tau_max:, ],
            # "activity": self.act[self.tau_max:, ],
        }, self.sbp_base, self.dbp_base, self.hr_base, self.rr_base, self.bt_base, self.spo_base, self.eda_base

    def plot(self, data):
        
        for i in range(data.shape[1]):
            plt.figure(figsize=(70, 10))
#             plt.subplot(data.shape[1], 1, i + 1)
#             plt.xlabel('time unit', fontsize=40)
            plt.title(data.columns[i], fontsize=60)
            plt.plot(data.iloc[:, i].values, color=plt.cm.Paired(i), linewidth=3)
            plt.rc('xtick', labelsize=40) 
            plt.rc('ytick', labelsize=40) 
            plt.savefig(str(data.columns[i]) +'.pdf')
            print(i)
        plt.show()


def indic(x, crit):
    if x >= crit:
        return 1
    else:
        return 0


def sig(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    if x < 0:
        return 0
    else:
        return x


def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def ar(x: list, t, lag):
    total = []
    for i in range(t - lag, t):
        total.append(x[i])
    return np.mean(total)
