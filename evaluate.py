#######################################Evaluate#####################################
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
import torchaudio
from DCUnet import DCUnet10

def evaluate(wav_path, path_to_model, N_FFT, hop_length):
    net = DCUnet10(N_FFT, hop_length)
    net.load_state_dict(torch.load(path_to_model))
    inp_wav = torchaudio.load(wav_path)
    out_wav = net(inp_wav, n_fft=N_FFT, hop_length=hop_length)
    # plot fre-time 
    signal = np.array(out_wav)
    origin = np.array(inp_wav)
    spectrogram = librosa.amplitude_to_db(librosa.stft(signal))
    spectrogram1 = librosa.amplitude_to_db(librosa.stft(origin))
    librosa.display.specshow(spectrogram, y_axis='log')
    librosa.display.specshow(spectrogram1, y_axis='log')

    # plot spectrum
    plt.figure(dpi=600) # 将显示的所有图分辨率调高

    plt.subplot(2,1,1)
    matplotlib.rc("font",family='SimHei') # 显示中文
    matplotlib.rcParams['axes.unicode_minus']=False # 显示符号
    plt.colorbar(format='%+2.0f dB')
    plt.title('去噪信号的对数谱图')
    plt.xlabel('时长（秒）')
    plt.ylabel('频率（赫兹）')
    plt.subplot(2,1,2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('源信号的对数谱图')
    plt.xlabel('时长（秒）')
    plt.ylabel('频率（赫兹）')
    plt.show()


