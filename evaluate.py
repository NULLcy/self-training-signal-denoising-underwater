#######################################Evaluate#####################################
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
from DCUnet import DCUnet10

def evaluate(wav, path_to_model, N_FFT, hop_length):
    net = DCUnet10(N_FFT, hop_length)
    net.load_state_dict(torch.load(path_to_model))
    out_wav = net(wav, n_fft=N_FFT, hop_length=hop_length)
    # plot fre-time 
    signal = np.array(out_wav)
    spectrogram = librosa.amplitude_to_db(librosa.stft(signal))
    librosa.display.specshow(spectrogram, y_axis='log')

    # plot spectrum
    plt.figure(dpi=600) # 将显示的所有图分辨率调高
    matplotlib.rc("font",family='SimHei') # 显示中文
    matplotlib.rcParams['axes.unicode_minus']=False # 显示符号
    plt.colorbar(format='%+2.0f dB')
    plt.title('语音信号对数谱图')
    plt.xlabel('时长（秒）')
    plt.ylabel('频率（赫兹）')
    plt.show()


