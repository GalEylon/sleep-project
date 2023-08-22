import mne
import numpy as np
import pandas as pd
from utilities import project_data_dir
from mne.io import read_raw_edf
from mne.filter import filter_data
import re
from scipy.stats import kurtosis

class EEGSleep:
    def __init__(self, project_name):
        """
        Initialize the EEGSleep class.

        Parameters:
        - project_name (str): The name of the project associated with the EEG data.
        """
        self.project_name = project_name
        self.raw_data = None
        self.preprocessed_data = None
        self.file_path = None
        self.best_channel = None

    def preprocess_eeg_data(self, file_path, preload=True, l_freq=0.75, h_freq=20):
        """
        Preprocess EEG data using MNE-Python.

        Parameters:
        - file_path (str): The path to the EEG data file.
        - preload (bool): Whether to preload the data into memory.
        - l_freq/h_freq (int, optional): lower and upper pass-band edges.

        Returns:
        - preprocessed_data (mne.io.Raw): The preprocessed EEG data.
        - info data structure (mne.Info)
        
        """
        # Load EEG data using MNE-Python
        self.raw_data = mne.io.read_raw_edf(file_path, preload=preload)
        
        # select EEG channels based on project name
        if self.project_name == 'NSRR':
            selectedChannelsPat = re.compile(r'\w{3} \w\w-[M]\w')
            eegChan = list(filter(selectedChannelsPat.match,self.raw_data.info['ch_names']))
            self.raw_data.pick(eegChan)
        elif self.project_name == 'Dreem':
            eegChan = ['EEG F7-O1', 'EEG F8-O2', 'EEG F8-O1', 'EEG F7-O2']
            self.raw_data.pick(eegChan)

        
        # preprocessing: filtering the data
        self.preprocessed_data = self.raw_data.load_data().filter(l_freq=l_freq, h_freq=h_freq, verbose = True)
        
        # Epoch the data into 30-second windows
        self.events = mne.make_fixed_length_events(self.preprocessed_data, start=0, duration=30)
        self.epochs = mne.Epochs(self.preprocessed_data, self.events, tmin=0, tmax=30, baseline=None)
        # Get epochs as numpy array
        self.data = self.epochs.get_data(units = 'uV')
        return self.raw_data.info, self.data
    
    def compute_noise_matrices(self, preprocessed_data, info, sd_crt = 1.5):
        # insert code to compute all the noise matrices
        
        if self.project_name == 'NSRR':
            
            ### FFT
            # define parameters for fft
            winlen = info['sfreq'] * 4
            overlap = info['sfreq'] * 2
            hzL = np.linspace(0, info['sfreq'] / 2, int(winlen / 2) + 1)
            # initilize the fft dataset to the same size as the data, data but only up to the nyquist frequency
            fftWelch = np.zeros((preprocessed_data.shape[0],preprocessed_data.shape[1],len(hzL)))
            
            # compute fft by iterating through channels and epochs
            for eleci in range(preprocessed_data.shape[1]):
                for epochi in range(preprocessed_data.shape[0]):
                    epochData = preprocessed_data[epochi,eleci,:]
                    numFrames = int((len(epochData) - winlen) / overlap) + 1
                    
                    # perform spectral analysis using the Welch method (divide epoch into overlapping segments and then average the periodogram)
                    fftA = np.zeros(len(hzL))
                    for j in range(numFrames):
                        frameData = epochData[int(j * overlap) :int( j * overlap + winlen)]
                        fftTemp = np.fft.fft(np.hamming(winlen) * frameData) / winlen
                        fftA += 2 * np.abs(fftTemp[:len(hzL)])

                    fftWelch[epochi,eleci :] = fftA / numFrames
            
            #sum up frequancies in the 10 to 20 Hz range
            Nfreqs = [10, 20]
            freqsidx = np.searchsorted(hzL, Nfreqs)
            fftWelch_all = np.squeeze(np.sum(fftWelch[:, :,freqsidx[0]:freqsidx[-1] + 1],axis=2))
            
            ### Additional noise parameters
            
            var_epoch_all = np.var(preprocessed_data, axis=2)  # variance
            kurt_epoch_all = kurtosis(preprocessed_data, axis=2)  # kurtosis
            mobility_epoch_all = np.sqrt(np.var(np.diff(preprocessed_data, axis=2), axis=2)) / (var_epoch_all)
            complexity_epoch_all = np.sqrt(np.var(np.diff(np.diff(preprocessed_data, axis=2), axis=2), axis=2)) / np.var(np.diff(preprocessed_data, axis=2), axis=2)
            max_amp = np.max(np.abs(preprocessed_data), axis=2)
            diff_all = np.median(np.diff(preprocessed_data, axis=2), axis=2)
            
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
            
            #Initialize empty datasets for all noise features
            fft_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            kurt_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            var_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            mob_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            comp_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            amp_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            diff_mark_all=  np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            maxk_amp_all = np.zeros((preprocessed_data.shape[0], preprocessed_data.shape[1]))
            
            # mark noise based on X (X=sd_crt) sd deviation from median
            
            for eleci in range(median_var_all.shape[0]):
              fft_mark_all[:, eleci] = np.where(fftWelch_all[:,eleci] > sd_crt * sd_fft_all[eleci] + median_fft_all[eleci], 1, 0)
              kurt_mark_all[:, eleci] = np.where(kurt_epoch_all[:,eleci] > sd_crt * sd_kurt_all[eleci] + median_kurt_all[eleci], 1, 0)
              var_mark_all[:, eleci] = np.where(var_epoch_all[:,eleci] > sd_crt * sd_var_all[eleci] + median_var_all[eleci], 1, 0)
              mob_mark_all[:, eleci] = np.where(mobility_epoch_all[:,eleci] > sd_crt * sd_mob_all[eleci] + median_mob_all[eleci], 1, 0)
              comp_mark_all[:, eleci] = np.where(complexity_epoch_all[:,eleci] > sd_crt * sd_comp_all[eleci] + median_comp_all[eleci], 1, 0)
              amp_mark_all[:, eleci] = np.where(max_amp[:,eleci] > sd_crt * sd_amp_all[eleci] + median_amp_all[eleci], 1, 0)
              diff_mark_all[:, eleci] = np.where(diff_all[:,eleci] > sd_crt * std_diff_all[eleci] + median_diff_all[eleci], 1, 0)

              abs_data = np.abs(preprocessed_data)
              indices = np.argsort(abs_data, axis=2)[:, eleci, -10:]
              amp_threshold = 200
              min_amp_count = 10

              for epochi in range(abs_data.shape[0]):
                  amp_exceed_count = np.sum(abs_data[epochi, eleci, indices[epochi, :]] > amp_threshold)
                  maxk_amp_all[epochi, eleci] = np.where(amp_exceed_count >= min_amp_count, 1, 0)

        return fft_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all, amp_mark_all, diff_mark_all, maxk_amp_all
    
    def noise_summary(self,fft_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all, amp_mark_all, diff_mark_all, maxk_amp_all, info):
        """
        Create a NumPy array that quantifies the amount of noise in each channel instead of choosing best electrode
        ***applies to NSRR data!***
        """
        
        # # Concatenate arrays
        mark_all = np.concatenate((fft_mark_all[:, :, np.newaxis], kurt_mark_all[:, :, np.newaxis],
                              var_mark_all[:, :, np.newaxis], mob_mark_all[:, :, np.newaxis],
                              comp_mark_all[:, :, np.newaxis], amp_mark_all[:, :, np.newaxis],
                              diff_mark_all[:, :, np.newaxis],maxk_amp_all[:, :, np.newaxis]),  axis=2)
        
        mark_all_by_elec = np.any(mark_all, axis=2)
        # Sum ones
        mark_all_by_elec_sum = np.sum(mark_all_by_elec, axis=0)
        # set pandas df with channel name and noise count
        elecdf = pd.DataFrame({'name':info['ch_names'], 'n_noise':mark_all_by_elec_sum})
        
        mark_prob = (np.sum(mark_all, axis=2)/np.shape(mark_all)[2])
        df_pred = pd.DataFrame()
        for i in range(mark_all_by_elec.shape[1]):
            df_pred[f"{elecdf.iloc[i]['name']} - noise"] = (mark_prob[:,i])
        
        return elecdf, df_pred

    def choose_best_electrode(self, epochs, fft_mark_all, kurt_mark_all, var_mark_all,
                              mob_mark_all, comp_mark_all, amp_mark_all, diff_mark_all, maxk_amp_all):
        """
        Choose the best electrode based on multiple criteria.

        Parameters:
        - epochs (mne.Epochs): The EEG epochs.
        - fft_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all,
          amp_mark_all, diff_mark_all, maxk_amp_all (numpy.ndarray): Mark arrays.

        Returns:
        - best_channel (numpy.ndarray): Array of indices indicating the best channel for each epoch.
        - rejected_epochs (numpy.ndarray): Array indicating rejected epochs.
        - final_mark (numpy.ndarray): Final mark array.
        """
        # Concatenate all the arrays
        mark_all = np.concatenate((fft_mark_all[:, :, np.newaxis], kurt_mark_all[:, :, np.newaxis],
                                   var_mark_all[:, :, np.newaxis], mob_mark_all[:, :, np.newaxis],
                                   comp_mark_all[:, :, np.newaxis], amp_mark_all[:, :, np.newaxis],
                                   diff_mark_all[:, :, np.newaxis], maxk_amp_all[:, :, np.newaxis]), axis=2)

        # Find the number of channels and epochs
        num_epochs, num_channels, _ = mark_all.shape

        # Create an array to store the sum of ones for each channel
        noise_per_channel = np.sum(mark_all, axis=2)

        # Find the channel with the minimum number of ones (least noise)
        self.best_channel = np.argmin(noise_per_channel, axis=1)

        # Flag noisy epochs for the "best channel" for that epoch
        best_channel_mark = noise_per_channel[np.arange(num_epochs), best_channel] > 0

        # Create final_mark as a binary vector
        final_mark = best_channel_mark

        epochs.reject = np.logical_not(~final_mark)

        rejected_epochs = epochs.reject
        return self.best_channel, rejected_epochs, final_mark
