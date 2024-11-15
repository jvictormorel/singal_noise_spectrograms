import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
import librosa.display
import os

# finding the directory containing the ".py" file

python_file_dir = os.path.dirname(os.path.realpath(__file__))
clean_speech_path = os.path.join(python_file_dir, 'clean_speech.wav')
cafeteria_0dB_path = os.path.join(python_file_dir, 'speech_cafeteria_0dB.wav')
cafeteria_5dB_path = os.path.join(python_file_dir, 'speech_cafeteria_5dB.wav')
cafeteria_m5dB_path = os.path.join(python_file_dir, 'speech_cafeteria_m5dB.wav')

# find the sampling rate of signals

clean_speech, fs_clean_speech = librosa.load(clean_speech_path, sr=None)
cafeteria_0dB, fs_0dB = librosa.load(cafeteria_0dB_path, sr=None)
cafeteria_5dB, fs_5dB = librosa.load(cafeteria_5dB_path, sr=None)
cafeteria_m5dB, fs_m5dB = librosa.load(cafeteria_m5dB_path, sr=None)

# noise's spectograms

D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_speech)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_clean, x_axis='time', y_axis='linear', sr=fs_clean_speech, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Clean Speech Spectrogram')
plt.show()

D_0dB = librosa.amplitude_to_db(np.abs(librosa.stft(cafeteria_0dB)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_0dB, x_axis='time', y_axis='linear', sr=fs_0dB, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Speech Cafeteria 0dB Spectrogram')
plt.show()

D_5dB = librosa.amplitude_to_db(np.abs(librosa.stft(cafeteria_5dB)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_5dB, x_axis='time', y_axis='linear', sr=fs_5dB, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Speech Cafeteria 5dB Spectrogram')
plt.show()

D_m5dB = librosa.amplitude_to_db(np.abs(librosa.stft(cafeteria_m5dB)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_m5dB, x_axis='time', y_axis='linear', sr=fs_m5dB, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Speech Cafeteria m5dB Spectrogram')
plt.show()

# amplitude x time

time = np.linspace(0, len(clean_speech) / fs_clean_speech, num=len(clean_speech))
plt.figure(figsize=(12, 4))
plt.plot(time, clean_speech, color='blue')
plt.title('Clean Speech')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()

time = np.linspace(0, len(cafeteria_0dB) / fs_0dB, num=len(cafeteria_0dB))
plt.figure(figsize=(12, 4))
plt.plot(time, cafeteria_0dB, color='blue')
plt.title('Speech Cafeteria 0dB')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()

time = np.linspace(0, len(cafeteria_5dB) / fs_5dB, num=len(cafeteria_5dB))
plt.figure(figsize=(12, 4))
plt.plot(time, cafeteria_5dB, color='blue')
plt.title('Speech Cafeteria 5dB')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()

time = np.linspace(0, len(cafeteria_m5dB) / fs_m5dB, num=len(cafeteria_m5dB))
plt.figure(figsize=(12, 4))
plt.plot(time, cafeteria_m5dB, color='blue')
plt.title('Speech Cafeteria m5dB')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()