import pandas as pd
import re
import numpy as np
import mne
from mne.io import read_raw_edf
from mne.filter import filter_data
import matplotlib.pyplot as plt
import yasa
import os
import sys

# from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, pearsonr
# from scipy.signal import welch
# from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize

# from string import ascii_uppercase
# import seaborn as sn
# from sklearn.metrics import confusion_matrix

import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

# MainPath = '/mnt/home/geylon/ceph/data/NSRR/'
MainPath = '//mnt/home/geylon/ceph/data/NSRR/nchsdb/sleep_data/'
os.chdir(MainPath)

# Get list of EDF and tsv files
edf_files = [file for file in os.listdir(MainPath) if file.endswith('.edf')]
noisefiles = [f for f in os.listdir('/mnt/home/geylon/ceph/data/NSRR/Results/noiseprob/') if f.endswith('.csv')]
noiseNames = [f.replace('.csv','') for f in noisefiles]
edf_files2 = [f for f in edf_files if f.replace('.edf','') not in noiseNames]

# edf_files = [file.replace('.edf','') for file in os.listdir(MainPath) if file.endswith('.edf')]
# csv_files = [file.replace('.tsv','') for file in os.listdir(MainPath) if file.endswith('.tsv')]

eeg_ind = int(sys.argv[1])
file = edf_files2[eeg_ind -1]

# if file.replace('.edf','.csv') in os.listdir('/mnt/home/geylon/ceph/data/NSRR/Results/noise/'):
#     print(file + 'already processed')
#     sys.exit()


# load edf file
EEG = mne.io.read_raw_edf(file, preload=True)
selectedChannelsPat = re.compile(r'\w{3} \w\w-[M]\w')
eegChan = list(filter(selectedChannelsPat.match,EEG.info['ch_names']))
if len(eegChan) ==0:
    print(file + 'non-referenced channels')
    sys.exit()

EEG.pick(eegChan)
# EEG.pick(['EEG F3-M2','EEG F4-M1','EEG C3-M2','EEG C4-M1','EEG O1-M2','EEG O2-M1'])
EEG.filter(l_freq=0.75, h_freq=20, verbose = False)
events = mne.make_fixed_length_events(EEG, start=0, duration=30)
epochs = mne.Epochs(EEG, events, tmin=0, tmax=30, baseline=None)

# convert EEG epoched data to df
data = epochs.get_data(units = 'uV')

#% noise identification
# FFT
winlen = EEG.info['sfreq'] * 4
overlap = EEG.info['sfreq'] * 2
hzL = np.linspace(0, EEG.info['sfreq'] / 2, int(winlen / 2) + 1)
fftWelch = np.zeros((data.shape[0],data.shape[1],len(hzL)))

for eleci in range(data.shape[1]):
    hzL = np.linspace(0, EEG.info['sfreq'] / 2, int(np.floor(winlen / 2) + 1))

    for epochi in range(data.shape[0]):
        epochData = data[epochi,eleci,:]
        numFrames = int((len(epochData) - winlen) / overlap) + 1
        fftA = np.zeros(len(hzL))

        for j in range(numFrames):
            frameData = epochData[int(j * overlap) :int( j * overlap + winlen)]
            fftTemp = np.fft.fft(np.hamming(winlen) * frameData) / winlen
            fftA += 2 * np.abs(fftTemp[:len(hzL)])

        fftWelch[epochi,eleci :] = fftA / numFrames

#sum up frequancies in 10 to 20 hz
Nfreqs = [10, 20]
freqsidx = np.searchsorted(hzL, Nfreqs)
fftWelch_all = np.squeeze(np.sum(fftWelch[:, :,freqsidx[0]:freqsidx[-1] + 1],axis=2))

# additional features
var_epoch_all = np.var(data, axis=2)  # variance
kurt_epoch_all = kurtosis(data, axis=2)  # kurtosis
mobility_epoch_all = np.sqrt(np.var(np.diff(data, axis=2), axis=2)) / (var_epoch_all)
complexity_epoch_all = np.sqrt(np.var(np.diff(np.diff(data, axis=2), axis=2), axis=2)) / np.var(np.diff(data, axis=2), axis=2)
max_amp = np.max(np.abs(data), axis=2)
diff_all = np.median(np.diff(data, axis=2), axis=2)

# get stds and medians

median_var_all = np.median(var_epoch_all,axis = 0);
sd_var_all = np.std(var_epoch_all,axis = 0);

median_kurt_all = np.median(kurt_epoch_all,axis = 0);
sd_kurt_all = np.std(kurt_epoch_all,axis=0);

median_fft_all = np.median(fftWelch_all, axis = 0);
sd_fft_all = np.std(fftWelch_all, axis = 0);

median_amp_all = np.median(max_amp, axis=0);
sd_amp_all = np.std(max_amp, axis=0);

median_mob_all = np.median(mobility_epoch_all, axis=0);
sd_mob_all = np.std(mobility_epoch_all, axis=0);

median_comp_all = np.median(complexity_epoch_all, axis=0);
sd_comp_all = np.std(complexity_epoch_all, axis=0);

median_diff_all = np.median(diff_all, axis=0);
std_diff_all = np.std(diff_all,axis=0);

#initilize empty datasets for all noise features
fft_mark_all=  np.zeros((data.shape[0], data.shape[1]))
kurt_mark_all=  np.zeros((data.shape[0], data.shape[1]))
var_mark_all=  np.zeros((data.shape[0], data.shape[1]))
mob_mark_all=  np.zeros((data.shape[0], data.shape[1]))
comp_mark_all=  np.zeros((data.shape[0], data.shape[1]))
amp_mark_all=  np.zeros((data.shape[0], data.shape[1]))
diff_mark_all=  np.zeros((data.shape[0], data.shape[1]))
maxk_amp_all = np.zeros((data.shape[0], data.shape[1]))

# mark noise based on X (X=sd_crt) sd deviation from median
sd_crt = 1.5
for eleci in range(median_var_all.shape[0]):
  fft_mark_all[:, eleci] = np.where(fftWelch_all[:,eleci] > sd_crt * sd_fft_all[eleci] + median_fft_all[eleci], 1, 0)
  kurt_mark_all[:, eleci] = np.where(kurt_epoch_all[:,eleci] > sd_crt * sd_kurt_all[eleci] + median_kurt_all[eleci], 1, 0)
  var_mark_all[:, eleci] = np.where(var_epoch_all[:,eleci] > sd_crt * sd_var_all[eleci] + median_var_all[eleci], 1, 0)
  mob_mark_all[:, eleci] = np.where(mobility_epoch_all[:,eleci] > sd_crt * sd_mob_all[eleci] + median_mob_all[eleci], 1, 0)
  comp_mark_all[:, eleci] = np.where(complexity_epoch_all[:,eleci] > sd_crt * sd_comp_all[eleci] + median_comp_all[eleci], 1, 0)
  amp_mark_all[:, eleci] = np.where(max_amp[:,eleci] > sd_crt * sd_amp_all[eleci] + median_amp_all[eleci], 1, 0)
  diff_mark_all[:, eleci] = np.where(diff_all[:,eleci] > sd_crt * std_diff_all[eleci] + median_diff_all[eleci], 1, 0)

  abs_data = np.abs(data)
  indices = np.argsort(abs_data, axis=2)[:, eleci, -10:]
  amp_threshold = 200
  min_amp_count = 10

  for epochi in range(abs_data.shape[0]):
      amp_exceed_count = np.sum(abs_data[epochi, eleci, indices[epochi, :]] > amp_threshold)
      maxk_amp_all[epochi, eleci] = np.where(amp_exceed_count >= min_amp_count, 1, 0)

# # Concatenate all the arrays
mark_all = np.concatenate((fft_mark_all[:, :, np.newaxis], kurt_mark_all[:, :, np.newaxis],
                      var_mark_all[:, :, np.newaxis], mob_mark_all[:, :, np.newaxis],
                      comp_mark_all[:, :, np.newaxis], amp_mark_all[:, :, np.newaxis],
                      diff_mark_all[:, :, np.newaxis],maxk_amp_all[:, :, np.newaxis]),  axis=2)

# Find nonzero elements per electrode
mark_all_by_elec = np.any(mark_all, axis=2)
# Sum ones
mark_all_by_elec_sum = np.sum(mark_all_by_elec, axis=0)

# set df with electrode name and noise count
elecdf = pd.DataFrame({'name':EEG.info['ch_names'], 'n_noise':mark_all_by_elec_sum})

# # Find electrode with minimum ones
# good_elec_ind = elecdf.n_noise.idxmin()
# # Find Occipital electrode with minimum ones
# good_elec_ind_oc = elecdf[elecdf['name'].str.contains('O')].n_noise.idxmin()
# # get the name of the ideal electrodes
# good_elec_name = EEG.info['ch_names'][good_elec_ind]
# good_elec_oc_name = EEG.info['ch_names'][good_elec_ind_oc]

# compute correlation between electrodes
# correlations = np.full((data.shape[0], data.shape[1], data.shape[1]),np.nan)

# for i in range(data.shape[0]):
#   for j in range(data.shape[1]):
#       for k in range(j+1,data.shape[1]):
#             correlations[i, j, k] = np.corrcoef(data[i, j, :], data[i, k, :])[0, 1]

# corr_mark_all = np.zeros((data.shape[0]))
# for d in range(correlations.shape[0]):
#   if (sum(correlations[d,:]>0.95)>=1).any():
#       corr_mark_all[d]=1

# final_mark = mark_all_by_elec[:,good_elec_ind]

# epochs.reject = (final_mark | np.array(corr_mark_all, dtype=bool))

# remove noisy epochs from fft
# fftClean = np.copy(fftWelch)
# fftClean[epochs.reject,good_elec_ind] = np.nan
# fftClean = fftClean[:,good_elec_ind]
print('finished proccessing: '+file)

mark_prob = (np.sum(mark_all, axis=2)/np.shape(mark_all)[2])
df_pred = pd.DataFrame()
for i in range(mark_all_by_elec.shape[1]):
    # df_pred[f"{elecdf.iloc[i]['name']} - noise"] = (mark_all_by_elec.astype(int)[:,i] | np.array(corr_mark_all, dtype=bool))
    df_pred[f"{elecdf.iloc[i]['name']} - noise"] = (mark_prob[:,i])
    

df_pred.to_csv(f"/mnt/home/geylon/ceph/data/NSRR/Results/noiseprob/{file.replace('.edf','')}.csv")

