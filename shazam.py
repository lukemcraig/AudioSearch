import os
import timeit

import numpy as np
import scipy.signal
import scipy.ndimage.filters
import scipy.ndimage.measurements
import scipy.io.wavfile
import pandas as pd
import pymongo
import librosa
from mutagen.easyid3 import EasyID3

# TODO conditional imports
from shazam_plots import plot_recognition_rate, plot_spectrogram_and_peak_subplots, start_hist_subplots, \
    make_next_hist_subplot, show_hist_plot, plot_hist_of_stks, plot_show, plot_scatter_of_fingerprint_offsets

# TODO class
time_functions = False
time_add_noise = True & time_functions
time_find_spec_peaks = True & time_functions
time_get_target_zone = True & time_functions
time_query_peaks_for_target_zone = True & time_functions
time_n_repeats = 1000


def main(insert_into_database=True, do_plotting=False):
    client = get_client()
    fingerprints_collection = client.audioprintsDB.fingerprints
    songs_collection = client.audioprintsDB.songs
    directory = 'C:/Users\Luke\Downloads/Disasterpeace/'
    mp3_filepaths = []
    for filepath in os.listdir(directory):
        if filepath[-4:] != '.mp3':
            continue
        mp3_filepaths.append(filepath)

    if insert_into_database:
        insert_mp3s_into_database(directory, do_plotting, fingerprints_collection, mp3_filepaths, songs_collection)
    else:
        measure_performance_of_multiple_snrs_and_mp3s(directory, do_plotting, fingerprints_collection, mp3_filepaths,
                                                      songs_collection)
    return


def insert_mp3s_into_database(directory, do_plotting, fingerprints_collection, mp3_filepaths, songs_collection):
    for mp3_i, mp3_filepath in enumerate(mp3_filepaths):
        print(mp3_filepath)
        data, rate, metadata = load_audio_data(directory + mp3_filepath)

        fingerprints = get_fingerprints_from_audio(data, rate, do_plotting)
        insert_one_song_into_database(metadata, fingerprints, fingerprints_collection, songs_collection)
    return


def measure_performance_of_multiple_snrs_and_mp3s(directory, do_plotting, fingerprints_collection, mp3_filepaths,
                                                  songs_collection):
    # mp3_filepaths = [mp3_filepaths[i] for i in [1, 2, 6, 7, 9, 14, 16, 18, 31, 29, 36]]
    # mp3_filepaths = [mp3_filepaths[i] for i in [31, 36]]
    snrs_to_test = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
    # snrs_to_test = [-30, -15]
    performance_results = np.zeros((len(mp3_filepaths), len(snrs_to_test)), dtype=bool)
    for mp3_i, mp3_filepath in enumerate(mp3_filepaths):
        print(mp3_filepath)
        data, rate, metadata = load_audio_data(directory + mp3_filepath)
        data_subset = get_test_subset(data)

        for snr_i, snr_dbfs in enumerate(snrs_to_test):
            correct_match, predicted_song_id = add_noise_and_predict_one_clip(data_subset, do_plotting,
                                                                              fingerprints_collection, metadata,
                                                                              mp3_filepath, rate, snr_dbfs,
                                                                              songs_collection)
            performance_results[mp3_i, snr_i] = correct_match

    recognition_rate = performance_results.mean(axis=0) * 100.0
    if do_plotting or True:
        plot_recognition_rate(recognition_rate, snrs_to_test)
    return


def add_noise_and_predict_one_clip(data_subset, do_plotting, fingerprints_collection, metadata, mp3_filepath, rate,
                                   snr_dbfs, songs_collection):
    data_and_noise = add_noise(data_subset, desired_snr_db=snr_dbfs)
    if time_add_noise:
        avg_time_add_noise = time_a_function(lambda: add_noise(data_subset, desired_snr_db=snr_dbfs))
        print("add_noise() took", '{0:.2f}'.format(avg_time_add_noise * 1000), "ms")
    predicted_song_id, correct_match = predict_one_audio_clip(data_and_noise, do_plotting,
                                                              fingerprints_collection, metadata, mp3_filepath,
                                                              rate, songs_collection)
    return correct_match, predicted_song_id


def predict_one_audio_clip(data_and_noise, do_plotting, fingerprints_collection, metadata, mp3_filepath, rate,
                           songs_collection):
    fingerprints = get_fingerprints_from_audio(data_and_noise, rate, do_plotting)
    predicted_song_id, correct_match = try_to_match_clip_to_database(do_plotting, mp3_filepath, fingerprints,
                                                                     fingerprints_collection,
                                                                     metadata, songs_collection)
    return predicted_song_id, correct_match


def time_a_function(func_lambda):
    print("warning: timing a function. This will cause unnecessary slowdowns.")
    timer_add_noise = timeit.Timer(func_lambda)
    time_taken_add_noise = timer_add_noise.timeit(number=time_n_repeats)
    avg_time_add_noise = time_taken_add_noise / time_n_repeats
    return avg_time_add_noise


def get_fingerprints_from_audio(data, rate, do_plotting):
    Sxx, f, t = get_spectrogram(data, rate)
    f_step = np.median(f[1:-1] - f[:-2])
    t_step = np.median(t[1:-1] - t[:-2])
    peak_locations, max_filter, max_filter_size = find_spectrogram_peaks(Sxx, t_step)

    if time_find_spec_peaks:
        avg_time = time_a_function(lambda: find_spectrogram_peaks(Sxx, t_step))
        print("Sxx was ", Sxx.shape)
        print("find_spectrogram_peaks() took", '{0:.2f}'.format(avg_time * 1000), "ms")
    if do_plotting:
        plot_spectrogram_and_peak_subplots(Sxx, f, max_filter, max_filter_size, peak_locations, t)

    fingerprints = get_fingerprints_from_peaks(len(f) - 1, f_step, peak_locations, len(t) - 1, t_step)
    return fingerprints


def try_to_match_clip_to_database(do_plotting, filepath, fingerprints, fingerprints_collection, metadata,
                                  songs_collection):
    # print("querying song in database")
    song = {'artist': metadata['artist'], 'album': metadata['album'], 'title': metadata['title'],
            'track_length_s': metadata['track_length_s']}
    song_doc = songs_collection.find_one(song)
    if song_doc is None:
        raise Exception(filepath + "needs to be inserted into the DB first!")
    # print("querying database")
    df_fingerprint_matches = get_df_of_fingerprint_offsets(do_plotting, fingerprints, fingerprints_collection)

    index_set = set(df_fingerprint_matches.index)
    n_possible_songs = len(index_set)
    if n_possible_songs == 0:
        # there were no fingerprints found, so we return an incorrect match result
        return -1, False

    max_hist_song = get_the_most_likely_song_from_all_the_histograms(df_fingerprint_matches, do_plotting, index_set,
                                                                     n_possible_songs)
    # TODO false positives?
    correct_match = max_hist_song == song_doc['_id']
    print("correct_match=", correct_match)
    if do_plotting:
        show_hist_plot(max_hist_song, song_doc)
    return max_hist_song, correct_match


def get_the_most_likely_song_from_all_the_histograms(df_fingerprint_matches, do_plotting, index_set, n_possible_songs):
    if do_plotting:
        ax = start_hist_subplots(n_possible_songs)
    max_hist_peak = 0
    max_hist_song = None
    for i, song_id in enumerate(index_set):
        if do_plotting:
            make_next_hist_subplot(ax, i, n_possible_songs, song_id)
        stks_in_songID = df_fingerprint_matches.loc[song_id]
        # make a histogram with bin width of 1
        unique, unique_counts = np.unique(stks_in_songID.values, return_counts=True)
        unique_max = unique.max()
        unique_min = unique.min()
        hist = np.zeros(1 + unique_max - unique_min)
        hist[unique - unique_min] = unique_counts
        # smooth histogram for the sake of "clustered peak detection"
        filtered_hist = scipy.ndimage.filters.uniform_filter1d(hist, size=2, mode='constant')
        max_filtered_hist = max(filtered_hist)
        if max_filtered_hist > max_hist_peak:
            max_hist_peak = max_filtered_hist
            max_hist_song = song_id
        if do_plotting:
            plot_hist_of_stks(np.arange(unique_min, unique_max + 1), hist)
            # overlay the filtered histogram
            plot_hist_of_stks(np.arange(unique_min, unique_max + 1), filtered_hist, alpha=0.5)
    return max_hist_song


def get_df_of_fingerprint_offsets(do_plotting, fingerprints, fingerprints_collection):
    stks = []
    db_fp_song_ids = []
    db_fp_offsets = []
    local_fp_offsets = []
    for fingerprint_i, fingerprint in enumerate(fingerprints):
        cursor = fingerprints_collection.find({'hash': fingerprint['hash']}, projection={"_id": 0, "hash": 0})
        for db_fp in cursor:
            db_fp_song_id = db_fp['songID']
            db_fp_song_ids.append(db_fp_song_id)
            # print(db_fp_song_id)
            db_fp_offset = db_fp['offset']
            db_fp_offsets.append(db_fp_offset)

            local_fp_offset = fingerprint['offset']
            local_fp_offsets.append(local_fp_offset)

            if do_plotting:
                plot_scatter_of_fingerprint_offsets(fingerprint_i, db_fp_offset, db_fp_song_id, local_fp_offset,
                                                    len(fingerprints))

            stk = db_fp_offset - local_fp_offset
            stks.append(stk)
    if do_plotting:
        plot_show()
    df_fingerprint_matches = pd.DataFrame({
        "songID": db_fp_song_ids,
        "stk": stks
    })
    df_fingerprint_matches.set_index('songID', inplace=True)
    return df_fingerprint_matches


def insert_one_song_into_database(metadata, fingerprints, fingerprints_collection, songs_collection):
    print("querying song in database")
    song = {'artist': metadata['artist'], 'album': metadata['album'], 'title': metadata['title'],
            'track_length_s': metadata['track_length_s']}
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
    random = np.random.RandomState(42)
    random_start_time = random.randint(0, len(data) - subset_length)

    data = data[random_start_time:random_start_time + subset_length]
    return data


def add_noise(data, desired_snr_db):
    # TODO real noise audio
    noise = get_white_noise(data)

    rms_signal = get_rms_linear(data)
    rms_noise = get_rms_linear(noise)

    desired_snr_linear = db_to_linear(desired_snr_db)
    adjustment = rms_signal / (rms_noise * desired_snr_linear)
    white_noise_adjusted = noise * adjustment

    return data + white_noise_adjusted


def get_white_noise(data):
    random = np.random.RandomState(42)
    white_noise = (random.random_sample(len(data)) * 2) - 1
    return white_noise


def db_to_linear(db_values):
    return 10 ** (db_values / 20)


def get_rms_dbfs_aes17(data):
    rms_linear = get_rms_linear(data)
    rms_dbfs_aes17 = convert_to_db(rms_linear) + 3
    return rms_dbfs_aes17


def convert_to_db(linear_values):
    return 20 * np.log10(linear_values)


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


def get_fingerprints_from_peaks(f_max, f_step, peak_locations, t_max, t_step):
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

        zone_freq_start, zone_freq_end, zone_time_start, zone_time_end = get_target_zone_bounds(anchor_f, anchor_t,
                                                                                                f_max, t_max,
                                                                                                zone_f_size,
                                                                                                zone_t_offset,
                                                                                                zone_t_size)
        if time_get_target_zone:
            avg_time = time_a_function(lambda: get_target_zone_bounds(anchor_f, anchor_t, f_max, t_max, zone_f_size,
                                                                      zone_t_offset, zone_t_size))
            print("get_target_zone_bounds() took", '{0:.2f}'.format(avg_time * 1000), "ms")

        # TODO better way to check the zone (sweep line)
        paired_df_peak_locations = query_dataframe_for_peaks_in_target_zone(df_peak_locations, zone_freq_end,
                                                                            zone_freq_start, zone_time_end,
                                                                            zone_time_start)
        if time_query_peaks_for_target_zone:
            avg_time = time_a_function(lambda: query_dataframe_for_peaks_in_target_zone(df_peak_locations,
                                                                                        zone_freq_end,
                                                                                        zone_freq_start, zone_time_end,
                                                                                        zone_time_start))
            print("query_dataframe_for_peaks_in_target_zone() took", '{0:.2f}'.format(avg_time * 1000), "ms")

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


def query_dataframe_for_peaks_in_target_zone(df_peak_locations, zone_freq_end, zone_freq_start, zone_time_end,
                                             zone_time_start):
    time_index = (df_peak_locations['t'] <= zone_time_end) & (df_peak_locations['t'] >= zone_time_start)
    freq_index = (zone_freq_start <= df_peak_locations['f']) & (df_peak_locations['f'] <= zone_freq_end)
    zone_index = time_index & freq_index
    n_pairs = zone_index.sum()
    paired_df_peak_locations = df_peak_locations[zone_index]
    return paired_df_peak_locations


def get_target_zone_bounds(anchor_f, anchor_t, f_max, t_max, zone_f_size, zone_t_offset, zone_t_size):
    zone_time_start = anchor_t + zone_t_offset
    zone_time_end = min(t_max, zone_time_start + zone_t_size)
    zone_freq_start = max(0, anchor_f - (zone_f_size // 2))
    zone_freq_end = min(f_max, zone_freq_start + zone_f_size)
    if zone_freq_end == f_max:
        zone_freq_start = zone_freq_end - zone_f_size
    return zone_freq_start, zone_freq_end, zone_time_start, zone_time_end


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


if __name__ == '__main__':
    main()
