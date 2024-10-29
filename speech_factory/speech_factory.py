import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
import librosa.display

# high quality graphics

plt.rcParams['figure.dpi'] = 300

# find the sampling rate of signals

noise_babble, fs_babble = librosa.load('babble.wav', sr=None)

noise_cafeteria, fs_cafeteria = librosa.load('cafeteria.wav', sr=None)

noise_factory, fs_factory = librosa.load('factory.wav', sr=None)

audio_clean, fs_clean = librosa.load('clean_speech.wav', sr=None)

audio_signal_5, fs_5 = librosa.load('speech_babble_0db.wav', sr=None)

# noise's spectograms

D_babb = librosa.amplitude_to_db(np.abs(librosa.stft(noise_babble)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_babb, x_axis='time', y_axis='linear', sr=fs_babble, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Babble Spectrogram')
plt.show()

D_caft = librosa.amplitude_to_db(np.abs(librosa.stft(noise_cafeteria)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_caft, x_axis='time', y_axis='linear', sr=fs_cafeteria, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Cafeteria Spectrogram')
plt.show()

D_fact = librosa.amplitude_to_db(np.abs(librosa.stft(noise_factory)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_fact, x_axis='time', y_axis='linear', sr=fs_factory, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Factory Spectrogram')
plt.show()

# speech's spectograms

D_4 = librosa.amplitude_to_db(np.abs(librosa.stft(audio_signal_4)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_4, x_axis='time', y_axis='linear', sr=fs_4, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram speech')
plt.show()

D_5 = librosa.amplitude_to_db(np.abs(librosa.stft(audio_signal_5)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_5, x_axis='time', y_axis='linear', sr=fs_5, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram speech_babble')
plt.show()

# amplitude x time

time = np.linspace(0, len(audio_signal_1) / fs_1, num=len(audio_signal_1))
plt.figure(figsize=(12, 4))
plt.plot(time, audio_signal_1, color='blue')
plt.title('Sinal de Áudio no Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()







