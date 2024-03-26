# am going to load two wave audio files and crop them in time two 2 minutes each , concatinate them to create 6 minutes of playing
# then i will play the audio and displaay the frequency content
# then i will apply the FFT and display the frequency content
# using torchaudio
import numpy as n
import io
import os
import tarfile
import tempfile
import torchaudio
import torch

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio



def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show()

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    # limit frequency for proper visualization
    plt.ylim(0, 1000)
    plt.show()
    
# i need to plot fft using pytorch and matplotlib
def plot_fft(waveform, sample_rate, title="FFT"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        with torch.no_grad():
            spec = torch.fft.rfft(torch.tensor(waveform[c]))
            spec = spec.abs().log1p()
            numpy_spec = spec.numpy()
            # get the number of frames
            num_freqs = numpy_spec.shape[-1]
            freq_axis = torch.linspace(0, sample_rate / 2, num_freqs)
            axes[c].plot(freq_axis.numpy(), numpy_spec, linewidth=1)
            axes[c].grid(True)
            axes[c].set_xlabel("Frequency [Hz]")
            # limit frequency for proper visualization 20Hz to 60 Hz
            #axes[c].set_xlim(20, 60)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show()

def _hide_seek(obj):
    class _wrapper:
        def __init__(self, obj):
            self.obj = obj

# load audio file from disk using torchaudio
filename = "40HzTone.wav"
waveform, sample_rate = torchaudio.load(filename)
print("Shape of waveform: {}".format(waveform.size()))
#plot_waveform(waveform, sample_rate)
#plot_specgram(waveform, sample_rate, title="Spectrogram of original audio file")
#plot_fft(waveform, sample_rate, title="FFT of original audio file")
waveform.numpy().shape
#concatinate the audio file to create 2 minutes of audio
# create a silence audio file of 2 minutes
silence = torch.zeros_like(waveform)
silence_duration = 2 # 2 minutes
silence_num_frames = int(sample_rate * silence_duration)
silence[:, :silence_num_frames] = 0
#concatinate the audio file
new_waveform = torch.cat([silence, waveform,], dim=1)
# then concatinate the audio file to create 6 minutes of audio
new_waveform = torch.cat([new_waveform, new_waveform], dim=1)
new_waveform = torch.cat([new_waveform, new_waveform], dim=1)




#plot_waveform(new_waveform, sample_rate)

import sounddevice
import time
#chcek the shape of the new waveform
sounddevice.play(new_waveform[0,:], sample_rate)  # releases GIL
time.sleep(1)
