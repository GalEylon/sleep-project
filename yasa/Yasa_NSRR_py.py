import pandas as pd
import re
import numpy as np
import mne
from mne.io import read_raw_edf
from mne.filter import filter_data
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# import PyQt5
# import ipympl
import yasa
import os
import sys

from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, pearsonr
from scipy.signal import welch
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize

from string import ascii_uppercase
import seaborn as sn
from sklearn.metrics import confusion_matrix

MainPath = '/mnt/home/geylon/ceph/data/NSRR/'
os.chdir(MainPath)

# Get list of EDF and tsv files
edf_files = [file for file in os.listdir(MainPath) if file.endswith('.edf')]
csv_files = [file for file in os.listdir(MainPath) if file.endswith(r'.tsv')]

eeg_ind = int(sys.argv[1])
file = edf_files[eeg_ind -1]

# cm = {}

# load edf file
EEG = mne.io.read_raw_edf(file, preload=True)
EEG.pick(['EEG F3-M2','EEG F4-M1','EEG C3-M2','EEG C4-M1','EEG O1-M2','EEG O2-M1'])
EEG.filter(l_freq=0.75, h_freq=20)
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
fft_mark_all=kurt_mark_all=kurt_mark_all=var_mark_all=mob_mark_all=comp_mark_all=amp_mark_all=diff_mark_all=maxk_amp_all = np.zeros((data.shape[0], data.shape[1]))

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

# Find electrode with minimum ones
good_elec_ind = elecdf.n_noise.idxmin()
# Find Occipital electrode with minimum ones
good_elec_ind_oc = elecdf[elecdf['name'].str.contains('O')].n_noise.idxmin()
# get the name of the ideal electrodes
good_elec_name = EEG.info['ch_names'][good_elec_ind]
good_elec_oc_name = EEG.info['ch_names'][good_elec_ind_oc]

# compute correlation between electrodes
correlations = np.zeros((data.shape[0], data.shape[1], data.shape[1]))

for i in range(data.shape[0]):
  for j in range(data.shape[1]):
      for k in range(data.shape[1]):
          correlations[i, j, k] = np.corrcoef(data[i, j, :], data[i, k, :])[0, 1]

corr_mark_all = np.zeros((data.shape[0]))
for d in range(correlations.shape[0]):
  if (sum(correlations[d,:]>0.9)>1).any():
      corr_mark_all[d]=1

final_mark = mark_all_by_elec[:,good_elec_ind]
# epochs.reject = np.logical_not(~final_mark)
epochs.reject = (final_mark | np.array(corr_mark_all, dtype=bool))

# remove noisy epochs from fft
fftClean = np.copy(fftWelch)
fftClean[epochs.reject,good_elec_ind] = np.nan
fftClean = fftClean[:,good_elec_ind]
print('finished proccessing: '+file)

# load sleep stages
csv_file = file.replace('.edf','.tsv')
sleep_stages = pd.read_csv(csv_file, sep = '\t')
# get only sleep events
sleep_stages = sleep_stages[sleep_stages['description'].str.match('^Sleep.*')==True]
# insert epoch count
sleep_stages['epoch'] = (round(sleep_stages['onset']/30)).astype(int)
# drop irrelevant coulmns
sleep_stages = sleep_stages.drop(['onset','duration'], axis=1).reset_index(drop = True)
# create dataframe to fill missing epochs 
df_to_join = pd.DataFrame({'description':'Sleep stage W','epoch': np.arange(1,sleep_stages['epoch'][0])})
#concat dfs
stages = pd.concat([df_to_join, sleep_stages], ignore_index = True)

#Yasa
df_pred = pd.DataFrame({'Manual': stages['description']})
for i in range(mark_all_by_elec.shape[1]):
    sls = yasa.SleepStaging(EEG, eeg_name = EEG.info['ch_names'][i])
    y_pred = sls.predict()
    confidence = sls.predict_proba().max(1)
    
    df_pred[f"{elecdf.iloc[i]['name']} ({elecdf.iloc[i]['n_noise']})"] = y_pred

    # df_pred = pd.DataFrame({'Stage': y_pred, 'Confidence': confidence})
    # df_pred['Manual'] = stages['description']
    # df_pred['noise']=epochs.reject

df_pred.replace(regex={r'.*W.*':0,r'.*R.*':1,r'.*N1.*':2, r'.*N2.*':3, r'.*N3.*':4,r'.*?.*':9}, inplace=True)
df_pred.to_csv(f"Results/SleepStages/{file.replace('.edf','')}.csv")
print('saved sleep stages file for: '+file)

# sls = yasa.SleepStaging(EEG, eeg_name = good_elec_name)
# y_pred = sls.predict()
# confidence = sls.predict_proba().max(1)

# df_pred = pd.DataFrame({'Stage': y_pred, 'Confidence': confidence})
# df_pred['Manual'] = stages['description']
# df_pred['noise']=epochs.reject
# df_pred.replace(regex={r'.*W.*':0,r'.*R.*':1,r'.*N1.*':2, r'.*N2.*':3, r'.*N3.*':4}, inplace=True)

# y_test = df_pred.loc[(~df_pred['noise']) & (df_pred['Manual'] != 'Sleep stage ?'),'Manual']
# y_pred = df_pred.loc[(~df_pred['noise']) & (df_pred['Manual'] != 'Sleep stage ?'),'Stage']
# columns = ['%s' %(i) for i in list(['Wake', 'REM','N1', 'N2','N3'])[0:len(np.unique(y_test))]]
# confm = confusion_matrix(y_test.tolist(),y_pred.tolist())
# confm = confm.astype('float') / confm.sum(axis=1)[:, np.newaxis]
# df_cm = pd.DataFrame(confm, index=columns, columns=columns)
# # cm[file]=df_cm
# df_cm.to_csv(f"Results/ConfusionMatrix/cm_{file.replace('.edf','')}.csv")
