import os

import numpy as np
import scipy.signal
import scipy.ndimage.filters
import scipy.ndimage.measurements
import scipy.io.wavfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib import scale as mscale
import matplotlib.patches as patches
import matplotlib.ticker as plticker
from mutagen.easyid3 import EasyID3

from power_scale import PowerScale
import pandas as pd
import resampy
import pymongo
import librosa


def main(insert_into_database=False, do_plotting=False):
    query_database = not insert_into_database

    client = get_client()
    fingerprints_collection = client.audioprintsDB.fingerprints
    songs_collection = client.audioprintsDB.songs
    directory = 'C:/Users\Luke\Downloads/Disasterpeace/'
    mp3_filepaths = []
    for filepath in os.listdir(directory):
        if filepath[-4:] != '.mp3':
            continue
        mp3_filepaths.append(filepath)

    # mp3_filepaths = mp3_filepaths[0:2]

    snrs_to_test = [15, 12, 9, 6, 3, 0, -3, -9, -12, -15]
    performance_results = np.zeros((len(mp3_filepaths), len(snrs_to_test)), dtype=bool)

    for mp3_i, mp3_filepath in enumerate(mp3_filepaths):
        print(mp3_filepath)
        data, rate, metadata = load_audio_data(directory + mp3_filepath)

        if query_database:
            data = get_test_subset(data)

            for snr_i, snr_dbfs in enumerate(snrs_to_test):
                data_and_noise = add_noise(data, desired_snr_dbfs=snr_dbfs)
                fingerprints = get_fingerprints_from_audio(data_and_noise, rate, do_plotting)
                correct_match = try_to_match_clip_to_database(do_plotting, mp3_filepath, fingerprints,
                                                              fingerprints_collection,
                                                              metadata, songs_collection)
                performance_results[mp3_i, snr_i] = correct_match
            pass

        elif insert_into_database:
            fingerprints = get_fingerprints_from_audio(data, rate, do_plotting)
            insert_one_song_into_database(metadata, fingerprints, fingerprints_collection, songs_collection)

    if query_database:
        recognition_rate = performance_results.mean(axis=0) * 100.0
        plt.style.use('ggplot')
        plt.plot(snrs_to_test, recognition_rate)
        plt.xlabel('Signal to Noise Ratio (dBFS)')
        plt.ylabel('Songs Identified')
        # plt.gca().xaxis.set_major_formatter(plticker.FormatStrFormatter('%d dBFS'))
        plt.gca().yaxis.set_major_formatter(plticker.FormatStrFormatter('%d %%'))
        plt.show()
        print()
    return


def get_fingerprints_from_audio(data, rate, do_plotting):
    Sxx, f, t = get_spectrogram(data, rate)
    f_step = np.median(f[1:-1] - f[:-2])
    t_step = np.median(t[1:-1] - t[:-2])
    peak_locations, max_filter, max_filter_size = find_spectrogram_peaks(Sxx, t_step)
    if do_plotting:
        plot_spectrogram_and_peak_subplots(Sxx, f, max_filter, max_filter_size, peak_locations, t)
    fingerprints = get_fingerprints_from_peaks(f, f_step, peak_locations, t, t_step)
    return fingerprints


def plot_spectrogram_and_peak_subplots(Sxx, f, max_filter, max_filter_size, peak_locations, t):
    ax = plt.subplot(2, 3, 1)
    plt.title("1. Spectrogram")
    plot_spectrogram(Sxx, f, t)
    plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
    plt.title("2. Max Filtered")
    plot_grid_of_filter_size(max_filter_size)
    plot_spectrogram(max_filter, f, t)
    # rect = patches.Rectangle((0, 0), max_filter_size[1] * t_step, max_filter_size[0] * f_step, edgecolor="black")
    # plt.gca().add_patch(rect)
    # ylim = plt.ylim()
    # xlim = plt.xlim()
    # for i in range(23):
    #     plt.axvline(x=max_filter_size[0] * t_step * i)
    # for i in range(9):
    #     plt.axhline(y=max_filter_size[1] * f_step * i)
    # plt.ylim(ylim)
    # plt.xlim(xlim)
    plt.subplot(2, 3, 3, sharex=ax, sharey=ax)
    plt.title("3.(A) Max Filtered == Spectrogram")
    plot_spectrogram(max_filter, f, t)
    plot_grid_of_filter_size(max_filter_size)
    plot_spectrogram_peaks(peak_locations, f, t)
    plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
    plt.title("3.(B) Max Filtered == Spectrogram")
    plot_spectrogram(Sxx, f, t)
    plot_grid_of_filter_size(max_filter_size)
    plot_spectrogram_peaks(peak_locations, f, t)
    plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
    plt.title("3.(C) Max Filtered == Spectrogram")
    # plot_spectrogram(Sxx, f, t)
    plot_grid_of_filter_size(max_filter_size)
    plot_spectrogram_peaks(peak_locations, f, t)
    plt.xlim(0, 500)
    plt.ylim(0, 512)
    plt.show()
    return


def try_to_match_clip_to_database(do_plotting, filepath, fingerprints, fingerprints_collection, metadata,
                                  songs_collection):
    # print("querying song in database")
    song = {'artist': metadata['artist'], 'album': metadata['album'], 'title': metadata['title'],
            'length': metadata['track_length_s']}
    song_doc = songs_collection.find_one(song)
    if song_doc is None:
        raise Exception(filepath + "needs to be inserted into the DB first!")
    # ax = plt.subplot(2, 1, 1)
    # print("querying database")
    if do_plotting:
        viridis = cm.get_cmap('viridis', len(fingerprints)).colors
    stks = []
    db_fp_song_ids = []
    db_fp_offsets = []
    local_fp_offsets = []
    for color_index, fingerprint in enumerate(fingerprints):
        cursor = fingerprints_collection.find({'hash': fingerprint['hash']}, projection={"_id": 0, "hash": 0})
        # cursor_listed = list(cursor)
        # df_fingerprint_matches = pd.DataFrame(cursor_listed)
        for db_fp in cursor:
            db_fp_song_id = db_fp['songID']
            db_fp_song_ids.append(db_fp_song_id)
            # print(db_fp_song_id)
            db_fp_offset = db_fp['offset']
            db_fp_offsets.append(db_fp_offset)

            local_fp_offset = fingerprint['offset']
            local_fp_offsets.append(local_fp_offset)

            if do_plotting:
                plt.scatter(db_fp_offset, local_fp_offset, c=viridis[color_index])
                plt.text(db_fp_offset, local_fp_offset, db_fp_song_id)

            stk = db_fp_offset - local_fp_offset
            stks.append(stk)
    if do_plotting:
        plt.show()
    df_fingerprint_matches = pd.DataFrame({
        "songID": db_fp_song_ids,
        "stk": stks
    })
    df_fingerprint_matches.set_index('songID', inplace=True)
    index_set = set(df_fingerprint_matches.index)
    n_possible_songs = len(index_set)
    if n_possible_songs == 0:
        return False
    if do_plotting:
        ax = plt.subplot(n_possible_songs, 1, 1)
    max_hist_peak = 0
    max_hist_song = None
    for i, song_id in enumerate(index_set):
        if do_plotting:
            if i > 0:
                plt.subplot(n_possible_songs, 1, i + 1, sharey=ax)
            plt.title("song_id:" + str(song_id))
        stks_in_songID = df_fingerprint_matches.loc[song_id]
        # TODO clustering histogram?
        # unique, unique_counts = np.unique(stks_in_songID.values, return_counts=True)
        hist, bin_edges = np.histogram(stks_in_songID.values, bins='auto')
        if max(hist) > max_hist_peak:
            max_hist_peak = max(hist)
            max_hist_song = song_id
        if do_plotting:
            hist_mpl, bin_edges_mpl, patches = plt.hist(stks_in_songID.values, bins='auto', rwidth=.9)
            # plt.bar(unique, unique_counts)
    # TODO false positives?
    correct_match = max_hist_song == song_doc['_id']
    print("correct_match=", correct_match)
    if do_plotting:
        plt.suptitle("matching song id=" + str(max_hist_song) + ",correct song=" + str(song_doc['_id']))
        plt.tight_layout()
        plt.show()
    return correct_match


def insert_one_song_into_database(metadata, fingerprints, fingerprints_collection, songs_collection):
    print("querying song in database")
    song = {'artist': metadata['artist'], 'album': metadata['album'], 'title': metadata['title'],
            'length': metadata['track_length_s']}
    song_doc = songs_collection.find_one(song)
    if song_doc is None:
        print("inserting song into database")
        most_recent_song = songs_collection.find_one({}, sort=[(u"_id", -1)])
        if most_recent_song is not None:
            new_id = most_recent_song['_id'] + 1
        else:
            new_id = 0
        song['_id'] = new_id
        insert_song_result = songs_collection.insert_one(song)
        song_doc = songs_collection.find_one({"_id": insert_song_result.inserted_id})
    print("inserting fingerprints into database")
    for fingerprint in fingerprints:
        fingerprint['songID'] = song_doc['_id']
        try:
            fingerprints_collection.insert_one(fingerprint)
        except pymongo.errors.DuplicateKeyError:
            continue


def get_test_subset(data):
    # subset_length = np.random.randint(rate * 5, rate * 14)
    subset_length = 112000
    subset_length = min(len(data), subset_length)
    random_start_time = np.random.randint(0, len(data) - subset_length)
    data = data[random_start_time:random_start_time + subset_length]
    return data


def add_noise(data, desired_snr_dbfs):
    # TODO real noise audio
    white_noise = (np.random.random(len(data)) * 2) - 1
    rms_signal = get_rms_linear(data)
    rms_noise = get_rms_linear(white_noise)
    snr = rms_signal / rms_noise
    desired_nsr_dbfs = -desired_snr_dbfs
    desired_nsr_linear = dbfs_to_linear(desired_nsr_dbfs)
    white_noise_adjusted = white_noise * snr * desired_nsr_linear
    # TODO test this, it's not right:
    actual_snr_dbfs = convert_to_dbfs(rms_signal / get_rms_linear(white_noise_adjusted))
    data += white_noise_adjusted
    return data


def dbfs_to_linear(snr):
    return np.exp(snr / 20)


def get_rms_dbfs_aes17(data):
    rms_linear = get_rms_linear(data)
    rms_dbfs_aes17 = convert_to_dbfs(rms_linear) + 3
    return rms_dbfs_aes17


def convert_to_dbfs(rms_linear):
    return 20 * np.log10(rms_linear)


def get_rms_linear(data):
    return np.sqrt(np.mean(np.square(data)))


def get_client():
    print("getting client...")
    client = pymongo.MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=3)
    client.server_info()
    print("got client")
    return client


def load_audio_data(filepath):
    print("loading audio")
    data, rate = librosa.load(filepath, mono=True, sr=8000)
    assert rate == 8000
    mp3tags = EasyID3(filepath)
    metadata = {
        "artist": mp3tags['artist'][0],
        "album": mp3tags['album'][0],
        "title": mp3tags['title'][0],
        "track_length_s": len(data) / rate
    }
    return data, rate, metadata


def downsample_audio(data, rate):
    print("downsampling audio")
    downsampled_rate = 8000
    data = resampy.resample(data, rate, downsampled_rate)
    rate = downsampled_rate
    return data, rate


def crop_audio_time(data, rate):
    print("cropping audio")
    data = data[:rate * 14]
    return data


def get_spectrogram(data, rate):
    # print('get_spectrogram')
    nperseg = 1024
    noverlap = int(np.round(nperseg / 1.5))
    # TODO scaling?
    f, t, Sxx = scipy.signal.spectrogram(data, fs=rate, scaling='spectrum',
                                         mode='magnitude',
                                         window='hann',
                                         nperseg=nperseg,
                                         noverlap=noverlap)
    return Sxx, f, t


def find_spectrogram_peaks(Sxx, t_step, f_size_hz=500, t_size_sec=2):
    # print('find_spectrogram_peaks')
    max_f = 4000
    f_bins = Sxx.shape[0]
    f_per_bin = max_f / f_bins
    f_size = int(np.round(f_size_hz / f_per_bin))
    t_size = int(np.round(t_size_sec / t_step))

    max_filter = scipy.ndimage.filters.maximum_filter(Sxx, size=(f_size, t_size), mode='constant')
    peak_locations = np.argwhere(Sxx == max_filter)
    return peak_locations, max_filter, (t_size, f_size)


def get_fingerprints_from_peaks(f, f_step, peak_locations, t, t_step):
    # print("get_fingerprints_from_peaks")
    # TODO fan out factor
    fan_out_factor = 10
    zone_f_size = 1400 // f_step
    zone_t_size = 6 // t_step
    zone_t_offset = 1
    df_peak_locations = pd.DataFrame(peak_locations, columns=['f', 't'])
    # df_peak_locations['f'] = f[df_peak_locations['f']]
    # df_peak_locations['t'] = t[df_peak_locations['t']]
    # sweep line + bst
    # df_peak_locations.sort_values(by='t', ascending=False)
    fingerprints = []
    print("n_peaks=", len(df_peak_locations))
    for i, anchor in df_peak_locations.iterrows():
        # print(i, end=", ")
        anchor_t = anchor['t']
        anchor_f = anchor['f']

        zone_time_start = anchor_t + zone_t_offset
        zone_time_end = min(len(t) - 1, zone_time_start + zone_t_size)

        zone_freq_start = max(0, anchor_f - (zone_f_size // 2))
        zone_freq_end = min(len(f) - 1, zone_freq_start + zone_f_size)
        if zone_freq_end == len(f) - 1:
            zone_freq_start = zone_freq_end - zone_f_size

        # TODO better way to check the zone (sweep line)
        time_index = (df_peak_locations['t'] <= zone_time_end) & (df_peak_locations['t'] >= zone_time_start)
        freq_index = (zone_freq_start <= df_peak_locations['f']) & (df_peak_locations['f'] <= zone_freq_end)
        zone_index = time_index & freq_index
        n_pairs = zone_index.sum()
        paired_df_peak_locations = df_peak_locations[zone_index]

        for j, second_peak in paired_df_peak_locations.iterrows():
            # print("    ", j, "/", n_pairs)
            second_peak_f = second_peak['f']
            time_delta = second_peak['t'] - anchor_t
            combined_key = combine_parts_into_key(anchor_f, second_peak_f, time_delta)
            # print(combined_key)
            fingerprint = {'hash': int(combined_key), 'offset': int(anchor_t)}
            fingerprints.append(fingerprint)
    # df_fingerprints = pd.DataFrame(fingerprints)
    return fingerprints


def combine_parts_into_key(peak_f, second_peak_f, time_delta):
    peak_f = np.uint32(peak_f)
    second_peak_f = np.uint32(second_peak_f)
    time_delta = np.uint32(time_delta)

    first_part = np.left_shift(peak_f, np.uint32(20))
    second_part = np.left_shift(second_peak_f, np.uint32(10))
    combined_key = first_part + second_part + time_delta
    return combined_key


def decode_hash(key):
    # only keep the 10 least significant bits
    time_delta = np.bitwise_and(key, np.uint32(1023))
    # shift 10 bits and only keep the 10 least significant bits
    second_peak_f = np.bitwise_and(np.right_shift(key, np.uint32(10)), np.uint32(1023))
    # shift 20 bits
    peak_f = np.right_shift(key, np.uint32(20))
    return peak_f, second_peak_f, time_delta


def plot_spectrogram_peaks(peak_locations, f, t):
    # plt.scatter(t[peak_locations[:, 1]], f[peak_locations[:, 0]], marker="*", c="red")
    plt.scatter(peak_locations[:, 1], peak_locations[:, 0], marker="*", c="Tomato", edgecolor="Snow", linewidths=.5)
    return


def plot_spectrogram(Sxx, f, t, alpha=1.0):
    color_norm = colors.LogNorm(vmin=1 / (2 ** 20), vmax=1)
    # color_norm = colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max())
    # plt.pcolormesh(t, f, Sxx, norm=color_norm, cmap='Greys')
    # plt.pcolormesh(t, f, Sxx, norm=color_norm, alpha=alpha)
    # plt.pcolormesh(Sxx, alpha=alpha, cmap='gray')
    # plt.pcolormesh(Sxx, norm=color_norm, alpha=alpha, cmap='gray')
    plt.pcolormesh(Sxx, norm=color_norm, alpha=alpha)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    plt.ylabel('Frequency Bin Index')
    plt.xlabel('Time Segment Index')
    # plt.yscale('log')
    # mscale.register_scale(PowerScale)
    # plt.yscale('powerscale', power=.5)

    # plt.yticks(rotation=0)
    # plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    return


def plot_grid_of_filter_size(max_filter_size):
    plt.gca().xaxis.set_major_locator(plticker.MultipleLocator(base=max_filter_size[0]))
    plt.gca().yaxis.set_major_locator(plticker.MultipleLocator(base=max_filter_size[1]))
    plt.grid(True, color='Black')
    return


if __name__ == '__main__':
    main()
