import pandas as pd
import numpy as np
from synthetic_data.data_generator import sci
from synthetic_data.data_generator_no_act import sci_no_act


def generate_dummy_patient(tu=1, n_hrs=0.5, tau_max=4, plotting=True, save=False):
    sci_patient = sci(tu, n_hrs, tau_max)
    patient = pd.DataFrame(sci_patient.get_signals(2))
    if plotting:
        sci_patient.plot(patient)
    if save:
        patient.to_csv('sample_patient.csv', index=False)

    return patient


def generate_dummy_patient_no_act(tu=1, n_hrs=1, tau_max=4, plotting=False, save=False):
    sci_patient = sci_no_act(tu, n_hrs, tau_max)
    patient, sbp_base, dbp_base, hr_base, rr_base, bt_base, spo2_base, eda_base= sci_patient.get_signals(2)
    patient = pd.DataFrame(patient)
    if plotting:
        sci_patient.plot(patient)
    if save:
        patient.to_csv('sample_patient.csv', index=False)
    patient_norm = patient
    patient_norm['sbp'] = (patient['sbp']-sbp_base)/sbp_base
    patient_norm['dbp'] = (patient['dbp']-dbp_base)/dbp_base
    patient_norm['heart_rate'] = (patient['heart_rate']-hr_base)/hr_base
    patient_norm['resp_rate'] = (patient['resp_rate']-rr_base)/rr_base
    patient_norm['temp_skin'] = (patient['temp_skin']-bt_base)/bt_base
    patient_norm['spo2'] = (patient['spo2']-spo2_base)/spo2_base
    patient_norm['bld_vol'] = robust_normalize(patient['bld_vol'])
    patient_norm['eda'] = (patient['eda']-eda_base)/eda_base
    return patient, patient_norm




def robust_normalize(data):
    median = np.median(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    normalized_data = (data - median) / iqr
    return normalized_data


