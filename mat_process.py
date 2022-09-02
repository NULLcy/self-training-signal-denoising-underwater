import wave
# from scipy.io import loadmat
import numpy as np
import h5py
import os

# one minute to one wav file
def mat2wav(mat_path, wav_path):
    UnprocessedSignal1 = h5py.File(mat_path)
    key = UnprocessedSignal1.keys()  
    len_ = UnprocessedSignal1.shape[1]
    for i in range(len_ // 60 + 1):
        data = np.array(UnprocessedSignal1[key][:,60*i:60*(i+1)])

        framerate = 12500   #fr

        wave_data = data
        wave_data = wave_data.astype(np.short)

        f = wave.open(os.path.join(wav_path, mat_path.split('\\')[-1].split('.')[0] + '_{}.wav'.format(i)), "wb")
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(framerate)
        f.writeframes(wave_data.tobytes())

        f.close()

