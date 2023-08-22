import pandas as pd
import re
import numpy as np
import mne
from mne.io import read_raw_edf
from mne.filter import filter_data
import matplotlib.pyplot as plt
import yasa
from usleep_api import USleepAPI
import os
import sys

import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

# MainPath = '/mnt/home/geylon/ceph/data/NSRR/'
MainPath = '//mnt/home/geylon/ceph/data/NSRR/nchsdb/sleep_data/'
os.chdir(MainPath)

# Get list of EDF and tsv files
edf_files = [file for file in os.listdir(MainPath) if file.endswith('.edf')]
stagesfiles = [f for f in os.listdir('/mnt/home/geylon/ceph/data/NSRR/Results/SleepStagesComb/') if f.endswith('.csv')]
stagesNames = [f.replace('.csv','') for f in stagesfiles]
edf_files2 = [f for f in edf_files if f.replace('.edf','') not in stagesNames]

eeg_ind = int(sys.argv[1])
file = edf_files2[eeg_ind -1]

if file.replace('.edf','.csv') in os.listdir('/mnt/home/geylon/ceph/data/NSRR/Results/SleepStagesComb/'):
    print(eeg_ind)
    print(file + 'already processed')
    sys.exit()

# load edf file
EEG = mne.io.read_raw_edf(file, preload=True)
selectedChannelsPat = re.compile(r'\w{3} \w\w-[M]\w')
eegChan = list(filter(selectedChannelsPat.match,EEG.info['ch_names']))
EEG.pick(eegChan)
EEG.filter(l_freq=0.75, h_freq=20, verbose = False)

#Yasa
df_pred = pd.DataFrame()
for i in EEG.info['ch_names']:
    sls = yasa.SleepStaging(EEG, eeg_name = i)
    y_pred = sls.predict()
    confidence = sls.predict_proba().max(1)
    
    df_pred[i+'_Y'] = y_pred
    df_pred[i+'_Y_conf'] = sls.predict_proba().max(1).tolist()

df_pred.replace(regex={r'.*W.*':0,r'.*R.*':1,r'.*N1.*':2, r'.*N2.*':3, r'.*N3.*':4,r'.*?.*':9}, inplace=True)
    
# U-sleep

token='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2OTAzODU2NzgsImlhdCI6MTY5MDM0MjQ3OCwibmJmIjoxNjkwMzQyNDc4LCJpZGVudGl0eSI6IjE4OGI0YjQ2M2Q4YSJ9.teHsV0VfzAxLvJCJncY4yILUXtyH9ld0SBeLIqHSJxI'
api = USleepAPI(api_token =token)
# api.delete_all_sessions()
hypnogram, _ = api.quick_predict(
    input_file_path=file,
    anonymize_before_upload=False
)
df_pred['U'] = hypnogram['hypnogram']
df_pred.U.replace([4,2],[2,4], inplace = True)

df_pred.to_csv(f"/mnt/home/geylon/ceph/data/NSRR/Results/SleepStagesComb/{file.replace('.edf','')}.csv")
print('saved file '+file)
# api.delete_all_sessions()