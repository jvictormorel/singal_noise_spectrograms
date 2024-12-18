import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
import librosa.display
import os

# finding the directory containing the ".py" file

python_file_dir = os.path.dirname(os.path.realpath(__file__))
babble_path = os.path.join(python_file_dir, 'babble.wav')
cafeteria_path = os.path.join(python_file_dir, 'cafeteria.wav')
factory_path = os.path.join(python_file_dir, 'factory.wav')

# finding the sampling rate of signals

noise_babble, fs_babble = librosa.load(babble_path, sr=None)
noise_cafeteria, fs_cafeteria = librosa.load(cafeteria_path, sr=None)
noise_factory, fs_factory = librosa.load(factory_path, sr=None)

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

# amplitude x time

time = np.linspace(0, len(noise_babble) / fs_babble, num=len(noise_babble))
plt.figure(figsize=(12, 4))
plt.plot(time, noise_babble, color='blue')
plt.title('Babble')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()

time = np.linspace(0, len(noise_cafeteria) / fs_cafeteria, num=len(noise_cafeteria))
plt.figure(figsize=(12, 4))
plt.plot(time, noise_cafeteria, color='blue')
plt.title('Cafeteria')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()

time = np.linspace(0, len(noise_factory) / fs_factory, num=len(noise_factory))
plt.figure(figsize=(12, 4))
plt.plot(time, noise_factory, color='blue')
plt.title('Factory')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()