import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
import librosa.display

# manter alta qualidade dos gráficos plotados

plt.rcParams['figure.dpi'] = 300

# descobrir a frequência de amostragem dos sinais

audio_signal_1, fs_1 = librosa.load('babble.wav', sr=None)
print("Taxa de amostragem:", fs_1)

audio_signal_2, fs_2 = librosa.load('cafeteria.wav', sr=None)
print("Taxa de amostragem:", fs_2)

audio_signal_3, fs_3 = librosa.load('factory.wav', sr=None)
print("Taxa de amostragem:", fs_3)








