from cgitb import Hook
import os
import gc
import torch
import torchaudio
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import interpolate
from torch.utils.data import DataLoader

from dataset_utils import SignalDataset, subsample2
from DCUnet import DCUnet10
from loss import RegularizedLoss


# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()
if (train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

# If running on Cuda set these 2 for determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# train_style
train_style = "self_supervised"

###################################### Parameters of Speech processing ##################################
SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

noise_class = "Random"
basepath = str(noise_class)
fixedpath = '/home/abc/n2n/SNA-DF/DCUnet10_complex_TSTM_subsample2/'

os.makedirs(fixedpath + basepath,exist_ok=True)
os.makedirs(fixedpath + basepath+"/Weights",exist_ok=True)
respath = fixedpath + basepath + '/results.txt'

######################################## Metrics for evaluation #########################################
def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old = np.linspace(0, duration, original.shape[0])
        time_new = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original

def extract_overlapped_windows(x,nperseg,noverlap,window=None):
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result

def SNRseg(clean_speech, processed_speech,fs, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps

    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    MIN_SNR     = -10 # minimum SNR in dB
    MAX_SNR     =  35 # maximum SNR in dB

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech,winlength,winlength-skiprate,hannWin)
    
    signal_energy = np.power(clean_speech_framed,2).sum(-1)
    noise_energy = np.power(clean_speech_framed-processed_speech_framed,2).sum(-1)
    
    segmental_snr = 10*np.log10(signal_energy/(noise_energy+eps)+eps)
    segmental_snr[segmental_snr<MIN_SNR]=MIN_SNR
    segmental_snr[segmental_snr>MAX_SNR]=MAX_SNR
    segmental_snr=segmental_snr[:-1] # remove last frame -> not valid
    return np.mean(segmental_snr)

def snr(reference, test):  
    numerator = 0.0
    denominator = 0.0
    for i in range(len(reference)):
        numerator += reference[i]**2
        denominator += (reference[i] - test[i])**2
    return 10*np.log10(numerator/denominator)


######################################## TRAIN #########################################

def train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()
    train_ep_loss = 0.
    counter = 0

    for x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft in train_loader:
        # zero gradients
        net.zero_grad()

        # for base training (input---g1_stft, target---fg1_wav)
        g1_stft = g1_stft.to(DEVICE)
        fg1_wav = net(g1_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
      
        # for regularization loss (input---x_noisy_stft, target---fx_wav)
        with torch.no_grad():
            x_noisy_stft = x_noisy_stft.to(DEVICE)  
            fx_wav = net(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
            g1fx, g2fx = subsample2(fx_wav)
            g1fx, g2fx = g1fx.type(torch.FloatTensor), g2fx.type(torch.FloatTensor)

        # calculate loss
        g1_wav, fg1_wav, g2_wav, g1fx, g2fx = g1_wav.to(DEVICE), fg1_wav.to(DEVICE), g2_wav.to(DEVICE), g1fx.to(DEVICE), g2fx.to(DEVICE)
        loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item()
        counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss

def test_epoch(net, test_loader, loss_fn, use_net=True):
    net.eval()
    test_ep_loss = 0.
    counter = 0.

    with torch.no_grad():
        for x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft in test_loader:

            # for base training (input---g1_stft, target---fg1_wav)
            g1_stft = g1_stft.to(DEVICE)
            fg1_wav = net(g1_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)

            # for regularization loss (input---x_noisy_stft, target---fx_wav)
            x_noisy_stft= x_noisy_stft.to(DEVICE)
            fx_wav = net(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
            g1fx, g2fx = subsample2(fx_wav)
            g1fx, g2fx = g1fx.type(torch.FloatTensor), g2fx.type(torch.FloatTensor)

            # calculate loss
            g1_wav, fg1_wav, g2_wav, g1fx, g2fx = g1_wav.to(DEVICE), fg1_wav.to(DEVICE), g2_wav.to(DEVICE), g1fx.to(DEVICE), g2fx.to(DEVICE)
            loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()

            test_ep_loss += loss.item()
            counter += 1

        test_ep_loss /= counter

        print("Actual compute done...testing now")

        # testmet = getMetricsonLoader(test_loader, net, use_net)

        # clear cache
        gc.collect()
        torch.cuda.empty_cache()

        return test_ep_loss

def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):

        train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
        test_loss = 0
        scheduler.step()
        print("Saving model....")

        with torch.no_grad():
            test_loss, testmet = test_epoch(net, test_loader, loss_fn, use_net=True)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        with open(fixedpath + basepath + '/results.txt', "a") as f:
            f.write("Epoch :" + str(e + 1) + "\n" + str(testmet))
            f.write("\n")

        print("OPed to txt")

        torch.save(net.state_dict(), fixedpath + basepath + '/Weights/dc10_model_' + str(e + 1) + '.pth')
        torch.save(optimizer.state_dict(), fixedpath + basepath + '/Weights/dc10_opt_' + str(e + 1) + '.pth')

        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        print("Epoch: {}/{}...".format(e+1, epochs),
                     "Loss: {:.6f}...".format(train_loss),
                     "Test Loss: {:.6f}".format(test_loss))
    return train_loss, test_loss

######################################## SELF-TRAINING #########################################

def pcm2wav(filepath,pcm, sample_rate, channels_first):
    torchaudio.save(filepath, pcm.cpu(), sample_rate, channels_first)

def self_train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()
    train_ep_loss = 0.
    counter = 0

    for x_noisy_stft, g1_stft, g1_wav, x_clean in train_loader:
        # zero gradients
        net.zero_grad()

        # for base training (input---g1_stft, target---fg1_wav)
        g1_stft = g1_stft.to(DEVICE)
        fg1_wav = net(g1_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
      
        # for regularization loss (input---x_noisy_stft, target---fx_wav)
        with torch.no_grad():
            x_noisy_stft = x_noisy_stft.to(DEVICE)  
            fx_wav = net(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
            g1fx, g2fx = subsample2(fx_wav)
            g1fx, g2fx = g1fx.type(torch.FloatTensor), g2fx.type(torch.FloatTensor)

        # calculate loss
        g1_wav, fg1_wav, g1fx, g2fx = g1_wav.to(DEVICE), fg1_wav.to(DEVICE), g1fx.to(DEVICE), g2fx.to(DEVICE)
        loss = loss_fn(g1_wav, fg1_wav, x_clean, g1fx, g2fx)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item()
        counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss

def self_train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):

        train_loss = self_train_epoch(net, train_loader, loss_fn, optimizer)
        test_loss = 0
        scheduler.step()
        print("Saving model....")

        with torch.no_grad():
            test_loss, testmet = test_epoch(net, test_loader, loss_fn, use_net=True)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        with open(fixedpath + basepath + '/results.txt', "a") as f:
            f.write("Epoch :" + str(e + 1) + "\n" + str(testmet))
            f.write("\n")

        print("OPed to txt")

        torch.save(net.state_dict(), fixedpath + basepath + '/Weights/dc10_model_' + str(e + 1) + '.pth')
        torch.save(optimizer.state_dict(), fixedpath + basepath + '/Weights/dc10_opt_' + str(e + 1) + '.pth')

        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        print("Epoch: {}/{}...".format(e+1, epochs),
                     "Loss: {:.6f}...".format(train_loss),
                     "Test Loss: {:.6f}".format(test_loss))
    return train_loss, test_loss


def self_training(net, threshold, TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, NO_LABEL_TRAIN_DIR, test_noisy_files, loss_fn, optimizer, scheduler, epochs, max_round):
    for i in range(max_round):
        print("self-train:{}-th".format(i))
        # predict no label train dataset
        train_input_files = sorted(list(TRAIN_INPUT_DIR.rglob('*.wav')))
        train_target_files = sorted(list(TRAIN_TARGET_DIR.rglob('*.wav')))
        no_label_train_files = sorted(list(NO_LABEL_TRAIN_DIR.rglob('*.wav')))

        no_label_train_dataset = SignalDataset(no_label_train_files)
        no_label_train_loader = DataLoader(no_label_train_dataset, batch_size = 1, shuffle=False)
        
        with torch.no_grad:
            for j, x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft in enumerate(no_label_train_loader):
                g1_stft = g1_stft.to(DEVICE)
                fg1_wav = net(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
                x_noisy = torch.istft(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
                if SNRseg(fg1_wav, x_noisy,fs=12500, frameLen=0.03, overlap=0.75) > threshold:
                    train_input_files.append(no_label_train_files[j])
                    filepath = os.path.join(TRAIN_TARGET_DIR, str(no_label_train_files[j]).split("\\")[-1].split('.')[0] + "_addLabel.wav")
                    pcm2wav(filepath, fg1_wav, 12500, True)
                    train_target_files.append(filepath)
                else:
                    continue
        train_loader = DataLoader(SignalDataset(train_input_files, train_target_files, N_FFT, HOP_LENGTH,self_training=True), batch_size = 2, shuffle=True)
        test_loader = DataLoader(SignalDataset(test_noisy_files), batch_size = 1, shuffle=False)
        train_losses, test_losses = self_train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs)
        return train_losses, test_losses
        
        
if __name__ == "__main__":
    if train_style == "unsupervised":
        ######################################## unsupervised Train CONFI #########################################
        TRAIN_INPUT_DIR = Path('/home/abc/n2n/Datasets/WhiteNoise_Train_Input')
        TEST_NOISY_DIR = Path('/home/abc/n2n/Datasets/WhiteNoise_Test_Input')

        train_input_files = sorted(list(TRAIN_INPUT_DIR.rglob('*.wav')))
        test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))

        test_dataset = SignalDataset(test_noisy_files, N_FFT, HOP_LENGTH)
        train_dataset = SignalDataset(train_input_files, N_FFT, HOP_LENGTH)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)


        # clear cache
        gc.collect()
        torch.cuda.empty_cache()

        dcunet = DCUnet10(N_FFT, HOP_LENGTH).to(DEVICE)
        optimizer = torch.optim.Adam(dcunet.parameters())
        loss_fn = RegularizedLoss()
        loss_fn = loss_fn.to(DEVICE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        # specify paths and uncomment to resume training from a given point
        # model_checkpoint = torch.load(path_to_model)
        # opt_checkpoint = torch.load(path_to_opt)
        # dcunet20.load_state_dict(model_checkpoint)
        # optimizer.load_state_dict(opt_checkpoint)

        train_losses, test_losses = train(dcunet, train_loader, test_loader, loss_fn, optimizer, scheduler, 20)
    
    else:

        ######################################## Self Train CONFI #########################################
        TRAIN_INPUT_DIR = Path('/home/abc/n2n/Datasets/WhiteNoise_Train_Input')
        TRAIN_TARGET_DIR = Path('/home/abc/n2n/Datasets/WhiteNoise_Train_Target')
        NO_LABEL_TRAIN_DIR = Path('/home/abc/n2n/Datasets/WhiteNoise_Train_No_Label_Input')
        TEST_NOISY_DIR = Path('/home/abc/n2n/Datasets/WhiteNoise_Test_Input')

        test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))

        # clear cache
        gc.collect()
        torch.cuda.empty_cache()

        dcunet = DCUnet10(N_FFT, HOP_LENGTH).to(DEVICE)
        optimizer = torch.optim.Adam(dcunet.parameters())
        loss_fn = RegularizedLoss()
        loss_fn = loss_fn.to(DEVICE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        train_loss, test_loss = self_training(dcunet, threshold=0.8, TRAIN_INPUT_DIR=TRAIN_INPUT_DIR, 
                            TRAIN_TARGET_DIR=TRAIN_TARGET_DIR, 
                            NO_LABEL_TRAIN_DIR=NO_LABEL_TRAIN_DIR,
                            test_noisy_files=test_noisy_files,loss_fn=loss_fn, 
                            optimizer=optimizer, scheduler=scheduler,epochs=20, max_round=6)


        
