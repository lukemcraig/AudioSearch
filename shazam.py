import numpy as np
import scipy.signal
import scipy.ndimage.filters
import scipy.ndimage.measurements
import scipy.io.wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import scale as mscale
from power_scale import PowerScale
import resampy


def main():
    data, rate = load_audio_data()
    data = crop_audio_time(data, rate)

    data, rate = downsample_audio(data, rate)

    Sxx, f, t = get_spectrogram(data, rate)

    # max_filter = scipy.ndimage.filters.maximum_filter(Sxx, size=(10, 10))
    # maxima = scipy.ndimage.measurements.maximum_position(Sxx)
    peaks, properties = scipy.signal.find_peaks(Sxx.flatten())
    peaks_x, peaks_y = np.unravel_index(peaks, dims=Sxx.shape)
    # xx, yy = np.meshgrid(np.arange(0, len(f)), np.arange(0, len(t)))
    plot_spectrogram(Sxx, f, t)

    plt.scatter(t[peaks_y], f[peaks_x])

    plt.show()
    return


def plot_spectrogram(Sxx, f, t):
    color_norm = colors.LogNorm(vmin=1 / (2 ** 20), vmax=1)
    # color_norm = colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
    # plt.pcolormesh(t, f, Sxx, norm=color_norm, cmap='Greys')
    plt.pcolormesh(t, f, Sxx, norm=color_norm)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    # plt.yscale('log')
    # mscale.register_scale(PowerScale)
    # plt.yscale('powerscale', power=.5)
    plt.yticks(rotation=0)
    plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    return


def get_spectrogram(data, rate):
    nperseg = 1024
    noverlap = int(np.round(nperseg / 1.5))
    f, t, Sxx = scipy.signal.spectrogram(data, fs=rate, scaling='spectrum',
                                         mode='magnitude',
                                         window='hann',
                                         nperseg=nperseg,
                                         noverlap=noverlap)
    return Sxx, f, t


def crop_audio_time(data, rate):
    data = data[:rate * 14]
    return data


def downsample_audio(data, rate):
    downsampled_rate = 8000
    data = resampy.resample(data, rate, downsampled_rate)
    rate = downsampled_rate
    return data, rate


def load_audio_data():
    rate, data = scipy.io.wavfile.read('C:/Users\Luke\Downloads\surfing on a rocket.wav')
    data = data[:, 0]
    data = data / (2 ** 15)
    return data, rate


main()
