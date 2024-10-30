import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import librosa
import librosa.display

# find the sampling rate of signals

clean_speech, fs_clean_speech = librosa.load(r'C:\Users\jvmor\OneDrive\Área de Trabalho\códigos_PIBIC_2024_2025\singal_noise_spectrograms\speech_babble\clean_speech.wav', sr=None)

babble_0dB, fs_0dB = librosa.load(r'C:\Users\jvmor\OneDrive\Área de Trabalho\códigos_PIBIC_2024_2025\singal_noise_spectrograms\speech_babble\speech_babble_0dB.wav', sr=None)

babble_5dB, fs_5dB = librosa.load(r'C:\Users\jvmor\OneDrive\Área de Trabalho\códigos_PIBIC_2024_2025\singal_noise_spectrograms\speech_babble\speech_babble_5dB.wav', sr=None)

babble_m5dB, fs_m5dB = librosa.load(r'C:\Users\jvmor\OneDrive\Área de Trabalho\códigos_PIBIC_2024_2025\singal_noise_spectrograms\speech_babble\speech_babble_m5dB.wav', sr=None)

# noise's spectograms

D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean_speech)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_clean, x_axis='time', y_axis='linear', sr=fs_clean_speech, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Clean Speech Spectrogram')
plt.show()

D_0dB = librosa.amplitude_to_db(np.abs(librosa.stft(babble_0dB)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_0dB, x_axis='time', y_axis='linear', sr=fs_0dB, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Speech Babble 0dB Spectrogram')
plt.show()

D_5dB = librosa.amplitude_to_db(np.abs(librosa.stft(babble_5dB)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_5dB, x_axis='time', y_axis='linear', sr=fs_5dB, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Speech Babble 5dB Spectrogram')
plt.show()

D_m5dB = librosa.amplitude_to_db(np.abs(librosa.stft(babble_m5dB)))
plt.figure(figsize=(10, 4))
librosa.display.specshow(D_m5dB, x_axis='time', y_axis='linear', sr=fs_m5dB, cmap='jet')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(format='%+2.0f dB')
plt.title('Speech Babble m5dB Spectrogram')
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

time = np.linspace(0, len(babble_0dB) / fs_0dB, num=len(babble_0dB))
plt.figure(figsize=(12, 4))
plt.plot(time, babble_0dB, color='blue')
plt.title('Speech Babble 0dB')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()

time = np.linspace(0, len(babble_5dB) / fs_5dB, num=len(babble_5dB))
plt.figure(figsize=(12, 4))
plt.plot(time, babble_5dB, color='blue')
plt.title('Speech Babble 5dB')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()

time = np.linspace(0, len(babble_m5dB) / fs_m5dB, num=len(babble_m5dB))
plt.figure(figsize=(12, 4))
plt.plot(time, babble_m5dB, color='blue')
plt.title('Speech Babble m5dB')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, time[-1])  
plt.grid()
plt.show()