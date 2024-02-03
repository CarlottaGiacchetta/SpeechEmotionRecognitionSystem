import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from prettytable import PrettyTable 

def function_plot(x, y, tip, sr, title):
  if tip == "time":
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
  if tip == "freq":
    plt.plot(x, np.abs(y))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()# Load thedio file


def table (dataset):
  data_as_list = dataset.to_dict(orient='records')
  # Creating a PrettyTable
  table = PrettyTable(dataset.columns.tolist())
  for row in data_as_list:
      table.add_row(row.values())
  print(table[1:10])