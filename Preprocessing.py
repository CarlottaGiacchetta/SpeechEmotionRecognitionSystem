import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
from prettytable import PrettyTable 


def create0(path):
  diz = {"W":"Anger", "L":"Beredom", "E":"Disgust", "A":"Anxiety", "F":"Happyness", "T":"Sadness", "N":"Neutral"}
  file = os.listdir(path)
  label_list = []
  name_list = []
  #HO AGGIUNTO LA PARTE SULL'AUDIO COSI ABBIAMO UN COLONNA IN PIù CHE TIENE L'AUDIO IN FUNZIONE DEL TEMPO
  audio_list = []
  for name in file:
    #create label & name columns
    name = name.split(".")
    name = name[0]
    letter = str(name[-2])
    label_list.append(diz[letter])
    name_list.append(name)
    #create audio column
    audio_path = f'./wav0/{name}.wav'
    audio, sr = librosa.load(audio_path) #audio signal is a one-dimensional NumPy array that represents the amplitude of the audio signal over time.
    audio_list.append(audio)

  return name_list, label_list, audio_list, sr

def create1(path, name_list, label_list, audio_list):
    diz = {"01":"Neutral", "02":"Calm", "03":"Happyness", "04":"Sadness", "05":"Anger", "06":"Fearful", "07":"Disgust", "08": "Surprised"}
    file = os.listdir(path)
    #HO AGGIUNTO LA PARTE SULL'AUDIO COSI ABBIAMO UN COLONNA IN PIù CHE TIENE L'AUDIO IN FUNZIONE DEL TEMPO
    #for name in file:
      #create label & name columns
      #name = name.split(".")
      #name = name[0]
      #emotion = name.split("-")
      #if emotion[5] == "01":
        #emotion = emotion[2]
        #label_list.append(diz[emotion])
        #name_list.append(name)
        #create audio column
        #audio_path = f'./wav1/{name}.wav'
        #audio, sr = librosa.load(audio_path) #audio signal is a one-dimensional NumPy array that represents the amplitude of the audio signal over time.
        #audio_list.append(list(audio))
    
    for name in file:
        # create label & name columns
        name_parts = name.split(".")
        name = name_parts[0]
        emotion = name.split("-")
        
        # Check if the emotion list has enough elements
        if len(emotion) > 5 and emotion[5] == "01":
            emotion = emotion[2]
            label_list.append(diz.get(emotion, "Unknown"))  # Use get() to handle unknown emotions
            name_list.append(name)
            # create audio column
            audio_path = f'./wav1/{name}.wav'
            audio, sr = librosa.load(audio_path)
            audio_list.append(list(audio))

    #create the dataset 
    dataset = pd.DataFrame({"name": name_list, "label":label_list, "audio_temp":audio_list})
    return dataset, sr

def  create2(path, name_list, label_list, audio_list):
    diz = {"01": "Happy", "02": "Sad", "03": "Angry", "04": "Surprise", "05": "Neutral"}
    file = os.listdir(path)
    # HO AGGIUNTO LA PARTE SULL'AUDIO COSI ABBIAMO UN COLONNA IN PIÙ CHE TIENE L'AUDIO IN FUNZIONE DEL TEMPO
    for name in file:
        # create label & name columns
        name_parts = name.split(".")
        name = name_parts[0]
        emotion = name.split("-")
        
        # Check if the emotion list has enough elements
        if len(emotion) > 5 and emotion[5] == "01":
            emotion = emotion[2]
            label_list.append(diz.get(emotion, "Unknown"))  # Use get() to handle unknown emotions
            name_list.append(name)
            # create audio column
            audio_path = f'./wav2/{name}.wav'
            audio, sr = librosa.load(audio_path)
            audio_list.append(list(audio))

    #create the dataset 
    dataset = pd.DataFrame({"name": name_list, "label":label_list, "audio_temp":audio_list})
    return dataset, sr


 
def FFT(dataset, sr, audio_temp_list):
  audio_Y_list =[]
  audio_freq_list = []
  for audio in audio_temp_list:
    # Calculate the FFT
    n = len(audio)
    k = np.arange(n)
    T = n / sr
    frq = k / T  # two sides frequency range
    frq = frq[:n//2]  # one side frequency range
    Y = np.fft.fft(audio)/n  # Fourier Transform and normalization
    Y = Y[:n//2]
    audio_freq_list.append(frq)
    audio_Y_list.append(Y)
  return audio_Y_list,audio_freq_list

def apply_hamming_window(dataset):
  audio_data = dataset.audio_temp
  hamming_list = []
  for audio in audio_data:
    # Create a Hamming window of the same length as the audio signal
    window = np.hamming(len(audio))
    # Apply the Hamming window to the audio signal
    audio_data_hamming = audio * window
    hamming_list.append(audio_data_hamming)
  dataset['audio_W'] = hamming_list
  return dataset

#VAD
def vad(dataset):
  audio_data = dataset.audio_W  
  energy_list=[]
  # Set the frame size and hop length (ho scelyo quelle usali)
  frame_size = 1024
  hop_length = 512
  # Calculate the energy for each frame
  for signal in audio_data: 
   energy = librosa.feature.rms(y=signal, frame_length=frame_size, hop_length=hop_length)[0]
   energy_list.append(energy)
   # Set a threshold for voice activity detection (potevamo anche imporlo noi ma farlo cosi mi sembra più preciso)
   threshold_energy = np.mean(energy) * 1.5

  # Perform VAD 
   vad_segments = [] # Initialize an empty list to store the segments identified by VAD
   is_voice = False # Initialize a flag to track whether the current frame is part of a speech segment or not
   # Iterate through each frame in the 'energy' array
   for i in range(0, len(energy)):
       # Check if the energy of the current frame is greater than a specified threshold (threshold_energy)
      if energy[i] > threshold_energy:
       # If the current frame's energy is above the threshold and it was not part of a speech segment,
       # mark the beginning of a new speech segment and append the corresponding time index to vad_segments
            if not is_voice:
                vad_segments.append(i * hop_length)
            is_voice = True
      else:
       # If the current frame's energy is below the threshold and it was part of a speech segment,
       # mark the end of the speech segment and append the corresponding time index to vad_segments
            if is_voice:
                vad_segments.append(i * hop_length)
            is_voice = False
  dataset['energy'] = energy_list

  #stop code if first audio totally silence
  total_silence = np.allclose(audio_data[0], 0, atol=1e-10)
  if total_silence:
    print("Total Silence")
    sys.exit()      

  # Plot the audio signal and VAD results in first audio 
  plt.plot(audio_data[0])
  plt.vlines(vad_segments, ymin=min(audio_data[0]), ymax=max(audio_data[0]), color='r', linestyle='dashed', label='VAD')
  plt.xlabel('Time (samples)')
  plt.ylabel('Amplitude')
  plt.title('Voice Activity Detection')
  plt.legend()
  plt.show()


#z-normalization
def z_normalize(dataset, control):
  if control == 'Frequency':
    audio_list=dataset.audio_wY
  if control == 'Time':
    audio_list=dataset.audio_W
  norm_audio=[]
  for signal in audio_list:
    mean = np.mean(signal)
    std_dev = np.std(signal)
    normalized_signal = (signal - mean) / std_dev
    norm_audio.append(normalized_signal)
  dataset[f'audio_N_{control}'] = norm_audio
  #print("\n\n PROVA DATASET NEL PREPROCESSING \n",dataset)
  return dataset
    

  


