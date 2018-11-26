import numpy as np
import scipy.signal
import scipy.ndimage.filters
import scipy.ndimage.measurements
import scipy.io.wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import scale as mscale
from power_scale import PowerScale
import pandas as pd
import resampy


def main():
    data, rate = load_audio_data()
    data = crop_audio_time(data, rate)

    data, rate = downsample_audio(data, rate)

    Sxx, f, t = get_spectrogram(data, rate)
    # plot_spectrogram(Sxx, f, t)

    t_step = np.median(t[1:-1] - t[:-2])
    peak_locations, max_filter = find_spectrogram_peaks(Sxx, t_step)

    plot_spectrogram(max_filter, f, t)
    plot_spectrogram_peaks(peak_locations, f, t)

    fan_out_factor = 10
    df_peak_locations = pd.DataFrame(peak_locations, columns=['f', 't'])
    # sweep line + bst
    df_peak_locations.sort_values(by='t', ascending=False)



    plt.ylim(0, 4000)
    plt.xlim(0, 14)
    plt.show()
    return


def plot_spectrogram_peaks(peak_locations, f, t):
    plt.scatter(t[peak_locations[:, 1]], f[peak_locations[:, 0]], marker="*", c="red")


def find_spectrogram_peaks(Sxx, t_step, f_size_hz=500, t_size_sec=2):
    max_f = 4000
    f_bins = Sxx.shape[0]
    f_per_bin = max_f / f_bins
    f_size = f_size_hz // f_per_bin
    t_size = int(np.round(t_size_sec / t_step))
    max_filter = scipy.ndimage.filters.maximum_filter(Sxx, size=(f_size, t_size))
    peak_locations = np.argwhere(Sxx == max_filter)
    return peak_locations, max_filter


def plot_spectrogram(Sxx, f, t, alpha=1.0):
    color_norm = colors.LogNorm(vmin=1 / (2 ** 20), vmax=1)
    # color_norm = colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
    # plt.pcolormesh(t, f, Sxx, norm=color_norm, cmap='Greys')
    plt.pcolormesh(t, f, Sxx, norm=color_norm, alpha=alpha)
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
