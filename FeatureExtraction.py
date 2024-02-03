import pandas as pd
import librosa
import librosa.display
import librosa.feature
from sklearn.decomposition import PCA
import numpy as np
from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import correlate, hilbert



def MFCCs(dataset, sample_rate):
  audio_temp_list = dataset.audio_N_Time
  #audio_temp_list = dataset.audio_temp
  MFCCS_list = []
  std_mfccs_list = []
  mean_mfccs_list = []
  for audio in audio_temp_list:
    #print("\n\n\n\n\n STAMPA SAMPLE RATE",sample_rate)
    features_mfcc = mfcc(audio, sample_rate)
    #print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
    #print('Length of each feature =', features_mfcc.shape[1])
    features_mfcc = features_mfcc.T
 
    MFCCS_list.append(features_mfcc)
    std_mfccs = np.std(features_mfcc, axis=0)
    mean_mfccs = np.mean(features_mfcc, axis=0)
    
    std_mfccs_list.append(std_mfccs)
    mean_mfccs_list.append(mean_mfccs)

  #dataset["MFCCS"] = MFCCS_list
  df_std = pd.DataFrame(std_mfccs_list)
  df_mean = pd.DataFrame(mean_mfccs_list) 
  return dataset, df_std, df_mean



def reduction(df, n_components, variance):
  if df.isnull().values.any():  
    df.fillna(0, inplace=True)
    print(df)
  pca = PCA(n_components=n_components)
  features = pca.fit_transform(df)
  print("\n\nvariance explained:\n",pca.explained_variance_ratio_)
  cumulative_vector = np.cumsum(pca.explained_variance_ratio_)
  print("\n\nvariance explained cumulative:\n",cumulative_vector)
  thereshold = cumulative_vector<variance
  thereshold = np.count_nonzero(thereshold)
  print(f"\n\nnumber of PCA that explain the {variance}% of variance of the data:",thereshold)
  features = features[:, :thereshold]
  features = pd.DataFrame(features)
  return features, thereshold

  

def LPCC(dataset):
  audio_temp_list = dataset.audio_N_Time
  LPCC_list = []
  std_lpcc_list = []

  for audio in audio_temp_list:
    features_lpcc = librosa.lpc(audio, order=13)
    #features_lpccs = features_lpccs.T
    #print('\nfeatures_lpcc =',features_lpcc)
    LPCC_list.append(features_lpcc)
  df_lpcc = pd.DataFrame(LPCC_list) 
  df_lpcc  = df_lpcc.iloc[:, 1:14]
  for i in range(13):
    df_lpcc.rename(columns={df_lpcc.columns[i]: f"lpcc_coeff_{i}"}, inplace=True)
  #dataset["LPCC"] = LPCC_list
  return dataset, df_lpcc



def calculate_intensity(dataset):
    Y = dataset.audio_N_Frequency
    int_db = []
    for signal in Y:
        # Calculate the magnitude of the signal
        magnitude = np.abs(signal)
        
        # Calculate the Root Mean Square (RMS) amplitude
        rms_amplitude = np.sqrt(np.mean(np.square(magnitude)))
        
        # Convert the RMS amplitude to decibels (dB)
        intensity_db = 20 * np.log10(rms_amplitude)
        
        int_db.append(intensity_db)
    
    dataset['intensity_db'] = int_db
    return dataset



def autocorrelation_pitch(dataset, sr):
    Y=dataset.audio_N_Time
    auto_pitch=[]
    for signal in Y:
     # Calculate the autocorrelation of the signal
      autocorr = np.correlate(signal, signal, mode='full')
     # Keep only the positive part of the autocorrelation (lags >= 0)
      autocorr = autocorr[len(autocorr)//2:]
     # Set a maximum lag to avoid picking up very low-frequency components
      max_lag = int(sr / 50)  # Assuming pitch range is around 50 Hz
      #print("MAX_LAG",max_lag)
      #print("AUTOCORR",autocorr[:max_lag])
      autocorr_max = autocorr[:max_lag]
     # Find the index of the maximum peak in the autocorrelation
      peak_index = np.max(autocorr_max)
     # Calculate the pitch in Hertz
      pitch = sr / peak_index
      auto_pitch.append(pitch)
      print(pitch)
    dataset['pitch'] = auto_pitch
    return dataset



#TEO stuff
#The Teager Energy Operator (TEO) is a mathematical tool used in signal processing to extract features from signals. 
# It is particularly useful for analyzing non-stationary signals, such as speech or music. The Teager Energy Operator is designed 
# to enhance the representation of energy changes in a signal and is often used in feature extraction for tasks like speech recognition 
# and audio signal processing.

#Computes a Teager energy operator (TEO) element for the input signal.
def teo(signal):
    teo_elem = signal**2 - np.roll(signal, 1) * np.roll(signal, -1)
    return teo_elem

#Calculates the auto-correlation of a given TEO signal.
def calculate_auto_correlation(teo_signal):
    cor = correlate(teo_signal, teo_signal, mode='full')
    return cor

#Computes the analytic signal envelope using the Hilbert transform.
def extract_envelope(signal):
    analytic_signal = hilbert(signal)
    return analytic_signal

#Calculates the area under the envelope curve using the trapezoidal rule.
def calculate_area(envelope):
    area = np.trapz(envelope)
    return area

# Normalizes the calculated area by dividing it by a normalization factor.
def normalize_area(area, normalization_factor):
    return area / normalization_factor

#Applies the above functions to a dataset of audio signals, computing TEO signals, 
#auto-correlations, envelopes, areas, and normalized areas.
def normalized_teo_auto_correlation_envelope_area(dataset):
    teo_signals = [teo(signal) for signal in dataset.audio_N_Time]
    auto_corr = [calculate_auto_correlation(teo_signal) for teo_signal in teo_signals]
    envelopes = [extract_envelope(signal) for signal in dataset.audio_N_Time]
    areas = [calculate_area(env) for env in envelopes]

    normalization_factors = [np.max(env) for env in envelopes]
    normalized_areas = [normalize_area(area, norm_fact) for area, norm_fact in zip(areas, normalization_factors)]
    dataset['norm_area'] = normalized_areas

    return dataset

#This function takes a TEO signal and the sr as inputs. 
#It computes the instantaneous frequency using the Hilbert transform and then calculates 
#the instantaneous phase and the corresponding instantaneous frequency.
def instantaneous_frequency(teo_signal, sr):
    analytic_signal = hilbert(teo_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    in_freq = np.diff(instantaneous_phase) / (2.0 * np.pi) * sr
    return in_freq

#This function decomposes the frequency modulation variation for a dataset. 
#It first computes TEO signals for each audio signal in the dataset, and then calculates the instantaneous 
#frequency using the instantaneous_frequency function. Finally, it computes the variance of 
#the instantaneous frequencies and adds a new column called 'teo_dec_freq_mod_var' to the dataset.
def teo_decomposed_frequency_modulation_variation(dataset, sampling_rate):
    teo_signals = [teo(signal) for signal in dataset.audio_N_Time]
    inst_freqs = [instantaneous_frequency(teo_signal, sampling_rate) for teo_signal in teo_signals]
    variations = [np.var(inst_freq) for inst_freq in inst_freqs]
    dataset['teo_dec_freq_mod_var'] = variations
    return dataset

#The function processes an audio dataset by applying the Teager Energy Operator, dividing the signal into critical bands, 
#calculating the auto-correlation and envelope for each band, computing the area under the envelope, normalizing the area, and accumulating the results. 
#The final result is a dataset with a new column representing the total normalized area across all critical bands.
def critical_band_teo_auto_correlation_envelope_area(dataset, sr):
    teo_signals = [teo(signal) for signal in dataset.audio_N_Time]

    # Define critical bands using librosa
    num_bands = 10
    critical_bands = [librosa.filtbank(signal, sr=sr, n_fft=2048, n_mels=num_bands) for signal in teo_signals]

    total_normalized_area = 0
    for band in critical_bands:
        auto_corr = calculate_auto_correlation(band)
        envelope = extract_envelope(auto_corr)
        area = calculate_area(envelope)

        # Choose an appropriate normalization factor for each band
        normalization_factor = np.max(envelope)

        normalized_area = normalize_area(area, normalization_factor)
        total_normalized_area += normalized_area
    
    dataset['cri_band_teo_auto_corr_env_a'] = total_normalized_area
    return dataset






