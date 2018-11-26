import numpy as np
import scipy.signal
import scipy.ndimage.filters
import scipy.ndimage.measurements
import scipy.io.wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib import scale as mscale
from mutagen.easyid3 import EasyID3

from power_scale import PowerScale
import pandas as pd
import resampy
from splay import SplayTree
import hashlib
import struct
import pymongo
import librosa


def main():
    client = get_client()
    fingerprints_collection = client.audioprintsDB.fingerprints
    songs_collection = client.audioprintsDB.songs

    data, rate, metadata = load_audio_data()
    # data = crop_audio_time(data, rate)

    # data, rate = downsample_audio(data, rate)

    Sxx, f, t = get_spectrogram(data, rate)
    # plot_spectrogram(Sxx, f, t)
    f_step = np.median(f[1:-1] - f[:-2])
    t_step = np.median(t[1:-1] - t[:-2])
    peak_locations, max_filter = find_spectrogram_peaks(Sxx, t_step)

    # plot_spectrogram(max_filter, f, t)
    # plot_spectrogram_peaks(peak_locations, f, t)

    fingerprints = get_fingerprints_from_peaks(f, f_step, peak_locations, t, t_step)

    query_database = False
    if query_database:
        print("querying database")
        viridis = cm.get_cmap('viridis', len(fingerprints)).colors
        stks = []
        for color_index, fingerprint in enumerate(fingerprints):
            cursor = fingerprints_collection.find({'hash': fingerprint['hash']})
            for db_fp in cursor:
                print(db_fp['songID'])
                db_fp_offset = db_fp['offset']
                local_fp_offset = fingerprint['offset']
                plt.scatter(db_fp_offset, local_fp_offset, c=viridis[color_index])
                stk = db_fp_offset - local_fp_offset
                stks.append(stk)
        plt.grid()
        plt.show()

        plt.hist(stks)
        plt.show()

    insert_into_database = True
    if insert_into_database:
        print("querying song in database")
        song = {'artist': metadata['artist'], 'album': metadata['album'], 'title': metadata['title'],
                'length': metadata['track_length_s']}
        song_doc = songs_collection.find_one(song)
        if song_doc is None:
            print("inserting song into database")
            new_id = songs_collection.find_one({}, sort=[(u"_id", -1)])['_id'] + 1
            song['_id'] = new_id
            insert_song_result = songs_collection.insert_one(song)
            song_doc = songs_collection.find_one({"_id": insert_song_result.inserted_id})
        print("inserting into database")
        for fingerprint in fingerprints:
            fingerprint['songID'] = song_doc['_id']
            try:
                fingerprints_collection.insert_one(fingerprint)
            except pymongo.errors.DuplicateKeyError:
                continue

    # plt.ylim(0, 4000)
    # plt.xlim(0, 14)
    # plt.show()
    return


def get_fingerprints_from_peaks(f, f_step, peak_locations, t, t_step):
    print("get_fingerprints_from_peaks")
    # TODO fan out factor
    fan_out_factor = 10
    zone_f_size = 1400 // f_step
    zone_t_size = 6 // t_step
    zone_t_offset = 1.5 // t_step
    df_peak_locations = pd.DataFrame(peak_locations, columns=['f', 't'])
    # df_peak_locations['f'] = f[df_peak_locations['f']]
    # df_peak_locations['t'] = t[df_peak_locations['t']]
    # sweep line + bst
    # df_peak_locations.sort_values(by='t', ascending=False)
    fingerprints = []
    for i, anchor in df_peak_locations.iterrows():
        print(i, "/", len(df_peak_locations))
        anchor_t = anchor['t']
        anchor_f = anchor['f']

        zone_time_start = anchor_t + zone_t_offset
        zone_time_end = min(len(t) - 1, zone_time_start + zone_t_size)

        zone_freq_start = max(0, anchor_f - (zone_f_size // 2))
        zone_freq_end = min(len(f) - 1, zone_freq_start + zone_f_size)
        if zone_freq_end == len(f) - 1:
            zone_freq_start = zone_freq_end - zone_f_size

        # TODO better way to check the zone
        time_index = (df_peak_locations['t'] <= zone_time_end) & (df_peak_locations['t'] >= zone_time_start)
        freq_index = (zone_freq_start <= df_peak_locations['f']) & (df_peak_locations['f'] <= zone_freq_end)
        zone_index = time_index & freq_index
        n_pairs = zone_index.sum()
        paired_df_peak_locations = df_peak_locations[zone_index]

        for j, second_peak in paired_df_peak_locations.iterrows():
            print("    ", j, "/", n_pairs)
            second_peak_f = second_peak['f']
            time_delta = second_peak['t'] - anchor_t
            combined_key = combine_parts_into_key(anchor_f, second_peak_f, time_delta)
            # print(combined_key)
            fingerprint = {'hash': int(combined_key), 'offset': int(anchor_t)}
            fingerprints.append(fingerprint)
    # df_fingerprints = pd.DataFrame(fingerprints)
    return fingerprints


def get_client():
    print("getting client...")
    client = pymongo.MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=3)
    client.server_info()
    print("got client")
    return client


def combine_parts_into_key(peak_f, second_peak_f, time_delta):
    combined_key = peak_f << 10
    combined_key += second_peak_f
    combined_key <<= 10
    combined_key += time_delta
    return combined_key


def plot_spectrogram_peaks(peak_locations, f, t):
    plt.scatter(t[peak_locations[:, 1]], f[peak_locations[:, 0]], marker="*", c="red")


def find_spectrogram_peaks(Sxx, t_step, f_size_hz=500, t_size_sec=2):
    print('find_spectrogram_peaks')
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
    print('get_spectrogram')
    nperseg = 1024
    noverlap = int(np.round(nperseg / 1.5))
    # TODO scaling?
    f, t, Sxx = scipy.signal.spectrogram(data, fs=rate, scaling='spectrum',
                                         mode='magnitude',
                                         window='hann',
                                         nperseg=nperseg,
                                         noverlap=noverlap)
    return Sxx, f, t


def crop_audio_time(data, rate):
    print("cropping audio")
    data = data[:rate * 14]
    return data


def downsample_audio(data, rate):
    print("downsampling audio")
    downsampled_rate = 8000
    data = resampy.resample(data, rate, downsampled_rate)
    rate = downsampled_rate
    return data, rate


def load_audio_data():
    print("loading audio")
    filepath = 'C:/Users\Luke\Downloads/Disasterpeace/Disasterpeace - Monsters Ate My Birthday Cake OST - 01 SECOND Stupidest Birthday Ever.mp3'
    data, rate = librosa.load(filepath, mono=True, sr=8000)
    assert rate == 8000
    mp3tags = EasyID3(filepath)
    metadata = {
        "artist": mp3tags['artist'][0],
        "album": mp3tags['album'][0],
        "title": mp3tags['title'][0],
        "track_length_s": len(data) / 8000
    }

    # rate, data = scipy.io.wavfile.read('C:/Users\Luke\Downloads/visitormiddle.wav')
    # left channel. TODO mono mixdown
    # data = data[:, 0]
    # for 16 bit audio
    # data = data / (2 ** 15)
    return data, rate, metadata


main()
