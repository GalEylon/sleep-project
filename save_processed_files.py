# This is the first step of the analysis:
# load edf file, choose electrodes, filter, extract epochs, save epochs as '.fif' file

import pandas as pd
import re
import numpy as np
import mne
from mne.io import read_raw_edf
from mne.filter import filter_data
import os
import sys

# change o the directory containing edf files
MainPath = '/mnt/home/geylon/ceph/data/NSRR/nchsdb/sleep_data/'
os.chdir(MainPath)

# Get list of EDF and tsv files
edf_files = [file for file in os.listdir(MainPath) if file.endswith('.edf')]

#choose file cased on slurm-task-id
eeg_ind = int(sys.argv[1])
file = edf_files[eeg_ind -1]

# # load edf file
EEG = mne.io.read_raw_edf(file, preload=True)
# pick only 'eeg', 'eog' and 'emg' channels
mne.pick_info(EEG.info, mne.pick_channels_regexp(EEG.info['ch_names'],'EEG|EOG|EMG'), copy = False)
# EEG.pick(['EEG F3-M2','EEG F4-M1','EEG C3-M2','EEG C4-M1','EEG O1-M2','EEG O2-M1',
#          'EOG LOC-M2', 'EOG ROC-M1', 'EMG CHIN1-CHIN2'])
#filter
EEG.filter(l_freq=0.75, h_freq=20)
#make epochs
events = mne.make_fixed_length_events(EEG, start=0, duration=30)
epochs = mne.Epochs(EEG, events, tmin=0, tmax=30, baseline=None)
# save epoch file
file_name = file.replace('.edf','-epo.fif')
epochs.save(f"/mnt/home/geylon/ceph/data/NSRR/nchsdb/processed_epoched/{file_name}", overwrite = True)