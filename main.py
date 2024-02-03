import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import correlate, hilbert


import Preprocessing as pre
import FeatureExtraction as ex
import Stampe as st

print_control = False #mi son rotta le palle di tutte le stampe, metti True se le vuoi vedere ahaha 
controller = False

for i in range(3):
    #print(i)
    path =  f"./wav{i}"
    if i == 0:
        name_list, label_list, audio_list, sr = pre.create0(path)
        #print(sr)
    else:
        if i == 1:
          dataset, sr = pre.create1(path, name_list, label_list, audio_list)
        #print(sr)
        elif i == 2:
         dataset, sr = pre.create2(path, name_list, label_list, audio_list)

    

print("\n",dataset)
print("\n",dataset.shape)
print("\n",dataset.tail())
print("\n",dataset.iloc[100])
print("\n",dataset.iloc[1000])


if controller == True:
    #FIRST STATISTICS ON THE VARIABLE LABEL
    print("\n\nAbsolute frequencies \n", dataset['label'].value_counts())
    print("\n\nRelative frequencies \n",dataset['label'].value_counts(normalize=True))





#TRY TO VISUALIZE THE FIRST AUDIO
audio = dataset['audio_temp']
audio = audio [0]
if print_control == True:
    st.function_plot(audio, '', "time", sr, "Waveform")



#PREPROCESSING OF THE AUDIO (FT TRASFORM)
audio_Y_list, audio_freq_list = pre.FFT(dataset, sr, dataset.audio_temp)
dataset['audio_Y'] = audio_Y_list
dataset['audio_freq'] = audio_freq_list

if controller == True: 
    print("\n\n\nPROVA FURIER TRASFROM\n")
    print("\n",dataset.head())

# Plot the frequency spectrum
freq = dataset.audio_freq
freq = freq[0]
print("\n\n\nFREQ", freq)
Y = dataset.audio_Y
Y = Y[0]
if print_control == True:
    st.function_plot(freq, Y, "freq", '', 'FFT of AudSignal')




#APPLY HAMMING WINDOW
dataset=pre.apply_hamming_window(dataset)
if controller == True: 
    print("\n\n\nPROVA HAMMING WINDOW\n")
    print("\n",dataset.head())
Y = dataset.audio_Y
Y = Y[0]
Z=dataset.audio_W
Z=[0]
freq = dataset.audio_freq
freq = freq[0]
audio = dataset.audio_W
audio = audio [0]

#plot the smoothed audio
if print_control == True:
    st.function_plot(audio, '', "time", sr, 'Waveform of Smoothed audio')
audio = dataset.audio_temp
audio = audio [0]

#plot the original audio
if print_control == True:
    st.function_plot(audio, '', "time", sr, 'Waveform of Original audio')





#PORTO IN FREQUENZA
audio_WY_list, audio_Wfreq_list = pre.FFT(dataset, sr, dataset.audio_W)
dataset['audio_wY'] = audio_WY_list
dataset['audio_wfreq'] = audio_Wfreq_list

if controller == True: 
    print("\n\n\nPROVA FURIER TRASFROM\n")
    print("\n",dataset.head())

# Plot the frequency spectrum of smoothed audio
freq = dataset.audio_wfreq
freq = freq[0]
if controller == True: 
    print("\n\n\nFREQ", freq)
Y = dataset.audio_wY
Y = Y[0]
if print_control == True:
    st.function_plot(freq, Y, "freq", '', 'FFT of Smoothed audio')

# Plot the frequency spectrum
freq = dataset.audio_freq
freq = freq[0]
if controller == True: 
    print("\n\n\nFREQ", freq)
Y = dataset.audio_Y
Y = Y[0]
if print_control == True:
    st.function_plot(freq, Y, "freq", '', 'FFT of Original audio')

print("\n",dataset)
print("\n",dataset.iloc[100])
#print("\n",dataset.iloc[1000])

#VAD
pre.vad(dataset)




#z-normalization
dataset = pre.z_normalize(dataset, control = 'Frequency') #control could be Time or Frequency
dataset = pre.z_normalize(dataset, control = 'Time')
if print_control == True:
    st.function_plot(freq, Y, "freq", '', 'FFT of Original audio')

if controller == True: 
    print("\n\n\nPROVA DATASET NORMALIZZATO\n", dataset)



print("\n",dataset)
print("\n",dataset.iloc[100])
print("\n",dataset.iloc[1000])


#FEATURES EXTRACTION



#MFCCs
#NB: sto facendo la features extraction prima di avere finito il preprocessing, ATTENZIONE poi bisognerà cambiare il dataset di imput dell funzione, per il resto sarà uguale
dataset, df_std, df_mean = ex.MFCCs(dataset, sr)

if controller == True: 
    print("\n\n\n PROVA FEATURES EXTRACTION MFCCs\n")
    print("\n\nstd\n", df_std)
    print(df_std.shape)
    print("\n\nmean\n", df_mean)
    print(df_mean.shape)

df_mfccs_std, thereshold_std = ex.reduction(df_std, n_components = 150, variance = 0.83)
if controller == True: 
    print("df_mfccs_std",df_mfccs_std)
df_mfccs_mean, thereshold_mean = ex.reduction(df_mean, n_components = 150, variance = 0.73)
if controller == True: 
    print("df_mfccs_mean",df_mfccs_mean)
for i in range(thereshold_std): #queste righe servono per creare il dataset con i nomi delle colonne
    df_mfccs_std.rename(columns={df_mfccs_std.columns[i]: f"mfccs_std_{i}"}, inplace=True)
for i in range(thereshold_mean):
    df_mfccs_mean.rename(columns={df_mfccs_mean.columns[i]: f"mfccs_mean_{i}"}, inplace=True)

if controller == True: 
    print("\n\n\n PROVA DATASET FEATURES EXTRACTION\n", df_mfccs_std)
    print("\n\n\n PROVA DATASET FEATURES EXTRACTION\n", df_mfccs_mean)


#LPCC
dataset, df_lpcc = ex.LPCC(dataset)

if controller == True: 
    print("\n\n\n PROVA FEATURES EXTRACTION LPCC\n")
    print("df_lpcc:\n", df_lpcc)
    print(df_lpcc.shape)

#Intensity
#MOLTE ZERO--->WHY? In alcuni casi, gli audio sono stati lavorarti per ridurre il rumore di fondo e migliorare 
#la qualità del segnale. 
#Questo potrebbe portare a una diminuzione dell'intensità misurata.
ex.calculate_intensity(dataset)
if controller == True:
    print(dataset.intensity_db)


dataset=ex.autocorrelation_pitch(dataset, sr)

dataset_end = pd.concat([dataset, df_mfccs_mean, df_mfccs_std, df_lpcc], axis = 1)#concatena i vari dataset

if controller == True: 
    print("quello che abbiamo chiamato dataset: \n", dataset)
    print("quello che abbiamo chiamato dataset_end columns: \n", dataset_end.columns)
    print("quello che abbiamo chiamato dataset_end intesity: \n", dataset_end.intensity_db)


#cose della teo
ex.normalized_teo_auto_correlation_envelope_area(dataset_end)
ex.teo_decomposed_frequency_modulation_variation(dataset_end, sr)

if controller == True: 
    print(dataset_end.norm_area)
    print(dataset_end.teo_dec_freq_mod_var)


dataset_end = dataset_end.drop(['audio_temp', 'audio_Y','audio_freq', 'audio_W',
       'audio_wY', 'audio_wfreq', 'audio_N_Frequency', 'norm_area','audio_N_Time', 'energy'], axis=1)

print("\n\n\n\n DATASET FINALE \n")
print(dataset_end)

print("\n\n\n\n DATASET FINALE colonne \n")
print(dataset_end.columns)

print("\n\n\n\n DATASET FINALE colonne \n")
print(dataset_end.shape)


print("\n",dataset_end.iloc[100])
print("\n",dataset_end.iloc[1000])


print("\n\n PROPORZIONI", dataset_end['label'].value_counts(normalize=True) * 100)

dataset_end.to_csv('dataset1.csv', index=False)