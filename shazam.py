import os
import timeit

import numpy as np
import scipy.signal
import scipy.ndimage.filters
import scipy.ndimage.measurements
import scipy.io.wavfile
import pandas as pd
import librosa
from mutagen.easyid3 import EasyID3

from audio_search_dbs import DuplicateKeyError
from mongo_audio_print_db import MongoAudioPrintDB
# TODO conditional imports
from shazam_plots import plot_recognition_rate, plot_spectrogram_and_peak_subplots, start_hist_subplots, \
    make_next_hist_subplot, show_hist_plot, plot_hist_of_stks, plot_show, plot_scatter_of_fingerprint_offsets


class AudioSearch:
    time_functions = False
    time_add_noise = True & time_functions
    time_find_spec_peaks = True & time_functions
    time_get_target_zone = True & time_functions
    time_query_peaks_for_target_zone = True & time_functions
    time_n_repeats = 1000

    def __init__(self, audio_prints_db, do_plotting=False):
        self.audio_prints_db = audio_prints_db
        self.do_plotting = do_plotting
        pass

    def insert_mp3s_fingerprints_into_database(self, mp3_filepaths):
        for mp3_i, mp3_filepath in enumerate(mp3_filepaths):
            print(mp3_filepath)
            data, rate, metadata = self.load_audio_data(mp3_filepath)

            fingerprints = self.get_fingerprints_from_audio(data, rate)
            self.insert_one_mp3_with_fingerprints_into_database(metadata, fingerprints)
        return

    def measure_performance_of_multiple_snrs_and_mp3s(self, mp3_filepaths):
        # mp3_filepaths = [mp3_filepaths[i] for i in [1, 2, 6, 7, 9, 14, 16, 18, 31, 29, 36]]
        # mp3_filepaths = [mp3_filepaths[i] for i in [31, 36]]
        snrs_to_test = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
        # snrs_to_test = [-30, -15]
        performance_results = np.zeros((len(mp3_filepaths), len(snrs_to_test)), dtype=bool)
        for mp3_i, mp3_filepath in enumerate(mp3_filepaths):
            print(mp3_filepath)
            data, rate, metadata = self.load_audio_data(mp3_filepath)
            data_subset = self.get_test_subset(data)

            for snr_i, snr_dbfs in enumerate(snrs_to_test):
                correct_match, predicted_song_id = self.add_noise_and_predict_one_clip(data_subset, metadata,
                                                                                       mp3_filepath, rate, snr_dbfs)
                performance_results[mp3_i, snr_i] = correct_match

        recognition_rate = performance_results.mean(axis=0) * 100.0
        if self.do_plotting:
            plot_recognition_rate(recognition_rate, snrs_to_test)
        return

    def add_noise_and_predict_one_clip(self, data_subset, metadata, mp3_filepath, rate, snr_dbfs):
        data_and_noise = self.add_noise(data_subset, desired_snr_db=snr_dbfs)
        if self.time_add_noise:
            avg_time_add_noise = self.time_a_function(lambda: self.add_noise(data_subset, desired_snr_db=snr_dbfs))
            print("add_noise() took", '{0:.2f}'.format(avg_time_add_noise * 1000), "ms")
        predicted_song_id, correct_match = self.predict_one_audio_clip(data_and_noise, metadata, mp3_filepath, rate)
        return correct_match, predicted_song_id

    def predict_one_audio_clip(self, data_and_noise, metadata, mp3_filepath, rate):
        fingerprints = self.get_fingerprints_from_audio(data_and_noise, rate)
        predicted_song_id, correct_match = self.try_to_match_clip_to_database(mp3_filepath, fingerprints, metadata)
        return predicted_song_id, correct_match

    def time_a_function(self, func_lambda):
        print("warning: timing a function. This will cause unnecessary slowdowns.")
        timer_add_noise = timeit.Timer(func_lambda)
        time_taken_add_noise = timer_add_noise.timeit(number=self.time_n_repeats)
        avg_time_add_noise = time_taken_add_noise / self.time_n_repeats
        return avg_time_add_noise

    def get_fingerprints_from_audio(self, data, rate):
        Sxx, f, t = self.get_spectrogram(data, rate)
        f_step = np.median(f[1:-1] - f[:-2])
        t_step = np.median(t[1:-1] - t[:-2])
        peak_locations, max_filter, max_filter_size = self.find_spectrogram_peaks(Sxx, t_step)

        if self.time_find_spec_peaks:
            avg_time = self.time_a_function(lambda: self.find_spectrogram_peaks(Sxx, t_step))
            print("Sxx was ", Sxx.shape)
            print("find_spectrogram_peaks() took", '{0:.2f}'.format(avg_time * 1000), "ms")
        if self.do_plotting:
            plot_spectrogram_and_peak_subplots(Sxx, f, max_filter, max_filter_size, peak_locations, t)

        fingerprints = self.get_fingerprints_from_peaks(len(f) - 1, f_step, peak_locations, len(t) - 1, t_step)
        return fingerprints

    def try_to_match_clip_to_database(self, filepath, fingerprints, metadata):
        # print("querying song in database")
        song = {'artist': metadata['artist'], 'album': metadata['album'], 'title': metadata['title'],
                'track_length_s': metadata['track_length_s']}
        song_doc = self.audio_prints_db.find_one_song(song)
        if song_doc is None:
            raise Exception(filepath + "needs to be inserted into the DB first!")
        # print("querying database")
        df_fingerprint_matches = self.get_df_of_fingerprint_offsets(fingerprints)

        index_set = set(df_fingerprint_matches.index)
        n_possible_songs = len(index_set)
        if n_possible_songs == 0:
            # there were no fingerprints found, so we return an incorrect match result
            return -1, False

        max_hist_song = self.get_the_most_likely_song_from_all_the_histograms(df_fingerprint_matches, index_set,
                                                                              n_possible_songs)
        # TODO false positives?
        correct_match = max_hist_song == song_doc['_id']
        print("correct_match=", correct_match)
        if self.do_plotting:
            show_hist_plot(max_hist_song, song_doc)
        return max_hist_song, correct_match

    def get_the_most_likely_song_from_all_the_histograms(self, df_fingerprint_matches, index_set, n_possible_songs):
        if self.do_plotting:
            ax = start_hist_subplots(n_possible_songs)
        max_hist_peak = 0
        max_hist_song = None
        for i, song_id in enumerate(index_set):
            if self.do_plotting:
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
            if self.do_plotting:
                plot_hist_of_stks(np.arange(unique_min, unique_max + 1), hist)
                # overlay the filtered histogram
                plot_hist_of_stks(np.arange(unique_min, unique_max + 1), filtered_hist, alpha=0.5)
        return max_hist_song

    def get_df_of_fingerprint_offsets(self, fingerprints):
        stks = []
        db_fp_song_ids = []
        db_fp_offsets = []
        local_fp_offsets = []
        for fingerprint_i, fingerprint in enumerate(fingerprints):
            db_fp_iterator = self.audio_prints_db.find_db_fingerprints_with_hash_key(fingerprint)
            for db_fp in db_fp_iterator:
                db_fp_song_id = db_fp['songID']
                db_fp_song_ids.append(db_fp_song_id)
                # print(db_fp_song_id)
                db_fp_offset = db_fp['offset']
                db_fp_offsets.append(db_fp_offset)

                local_fp_offset = fingerprint['offset']
                local_fp_offsets.append(local_fp_offset)

                if self.do_plotting:
                    plot_scatter_of_fingerprint_offsets(fingerprint_i, db_fp_offset, db_fp_song_id, local_fp_offset,
                                                        len(fingerprints))

                stk = db_fp_offset - local_fp_offset
                stks.append(stk)
        if self.do_plotting:
            plot_show()
        df_fingerprint_matches = pd.DataFrame({
            "songID": db_fp_song_ids,
            "stk": stks
        })
        df_fingerprint_matches.set_index('songID', inplace=True)
        return df_fingerprint_matches

    def insert_one_mp3_with_fingerprints_into_database(self, metadata, fingerprints):
        song_id_in_db = self.get_or_insert_song_into_db(metadata)
        print("inserting fingerprints into database")
        for fingerprint in fingerprints:
            fingerprint['songID'] = song_id_in_db
            try:
                self.audio_prints_db.insert_one_fingerprint(fingerprint)
            except DuplicateKeyError:
                # song already exists in db
                continue
        return

    def get_or_insert_song_into_db(self, metadata):
        print("querying song in database")
        song = {'artist': metadata['artist'], 'album': metadata['album'], 'title': metadata['title'],
                'track_length_s': metadata['track_length_s']}
        song_doc = self.audio_prints_db.find_one_song(song)
        if song_doc is None:
            print("inserting song into database")
            new_id = self.audio_prints_db.get_next_song_id()
            song['_id'] = new_id

            inserted_id = self.audio_prints_db.insert_one_song(song)
            return inserted_id
        else:
            return song_doc['_id']

    def get_test_subset(self, data):
        # subset_length = np.random.randint(rate * 5, rate * 14)
        subset_length = 112000
        subset_length = min(len(data), subset_length)
        random = np.random.RandomState(42)
        random_start_time = random.randint(0, len(data) - subset_length)

        data = data[random_start_time:random_start_time + subset_length]
        return data

    def add_noise(self, data, desired_snr_db):
        # TODO real noise audio
        noise = self.get_white_noise(data)

        rms_signal = self.get_rms_linear(data)
        rms_noise = self.get_rms_linear(noise)

        desired_snr_linear = self.db_to_linear(desired_snr_db)
        adjustment = rms_signal / (rms_noise * desired_snr_linear)
        white_noise_adjusted = noise * adjustment

        return data + white_noise_adjusted

    def get_white_noise(self, data):
        random = np.random.RandomState(42)
        white_noise = (random.random_sample(len(data)) * 2) - 1
        return white_noise

    def db_to_linear(self, db_values):
        return 10 ** (db_values / 20)

    def get_rms_dbfs_aes17(self, data):
        rms_linear = self.get_rms_linear(data)
        rms_dbfs_aes17 = self.convert_to_db(rms_linear) + 3
        return rms_dbfs_aes17

    def convert_to_db(self, linear_values):
        return 20 * np.log10(linear_values)

    def get_rms_linear(self, data):
        return np.sqrt(np.mean(np.square(data)))

    def load_audio_data(self, filepath):
        print("loading audio")
        desired_rate = 8000
        data, rate = librosa.load(filepath, mono=True, sr=desired_rate)
        assert rate == desired_rate
        mp3tags = EasyID3(filepath)
        metadata = {
            "artist": mp3tags['artist'][0],
            "album": mp3tags['album'][0],
            "title": mp3tags['title'][0],
            "track_length_s": len(data) / rate
        }
        return data, rate, metadata

    def get_spectrogram(self, data, rate):
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

    def find_spectrogram_peaks(self, Sxx, t_step, f_size_hz=500, t_size_sec=2):
        # print('find_spectrogram_peaks')
        max_f = 4000
        f_bins = Sxx.shape[0]
        f_per_bin = max_f / f_bins
        f_size = int(np.round(f_size_hz / f_per_bin))
        t_size = int(np.round(t_size_sec / t_step))

        max_filter = scipy.ndimage.filters.maximum_filter(Sxx, size=(f_size, t_size), mode='constant')
        peak_locations = np.argwhere(Sxx == max_filter)
        return peak_locations, max_filter, (t_size, f_size)

    def get_fingerprints_from_peaks(self, f_max, f_step, peak_locations, t_max, t_step):
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

            zone_freq_start, zone_freq_end, zone_time_start, zone_time_end = self.get_target_zone_bounds(anchor_f,
                                                                                                         anchor_t,
                                                                                                         f_max, t_max,
                                                                                                         zone_f_size,
                                                                                                         zone_t_offset,
                                                                                                         zone_t_size)
            if self.time_get_target_zone:
                avg_time = self.time_a_function(
                    lambda: self.get_target_zone_bounds(anchor_f, anchor_t, f_max, t_max, zone_f_size,
                                                        zone_t_offset, zone_t_size))
                print("get_target_zone_bounds() took", '{0:.2f}'.format(avg_time * 1000), "ms")

            # TODO better way to check the zone (sweep line)
            paired_df_peak_locations = self.query_dataframe_for_peaks_in_target_zone(df_peak_locations, zone_freq_end,
                                                                                     zone_freq_start, zone_time_end,
                                                                                     zone_time_start)
            if self.time_query_peaks_for_target_zone:
                avg_time = self.time_a_function(lambda: self.query_dataframe_for_peaks_in_target_zone(df_peak_locations,
                                                                                                      zone_freq_end,
                                                                                                      zone_freq_start,
                                                                                                      zone_time_end,
                                                                                                      zone_time_start))
                print("query_dataframe_for_peaks_in_target_zone() took", '{0:.2f}'.format(avg_time * 1000), "ms")

            for j, second_peak in paired_df_peak_locations.iterrows():
                # print("    ", j, "/", n_pairs)
                second_peak_f = second_peak['f']
                time_delta = second_peak['t'] - anchor_t
                combined_key = self.combine_parts_into_key(anchor_f, second_peak_f, time_delta)
                # print(combined_key)
                fingerprint = {'hash': int(combined_key), 'offset': int(anchor_t)}
                fingerprints.append(fingerprint)
        # df_fingerprints = pd.DataFrame(fingerprints)
        return fingerprints

    def query_dataframe_for_peaks_in_target_zone(self, df_peak_locations, zone_freq_end, zone_freq_start, zone_time_end,
                                                 zone_time_start):
        time_index = (df_peak_locations['t'] <= zone_time_end) & (df_peak_locations['t'] >= zone_time_start)
        freq_index = (zone_freq_start <= df_peak_locations['f']) & (df_peak_locations['f'] <= zone_freq_end)
        zone_index = time_index & freq_index
        n_pairs = zone_index.sum()
        paired_df_peak_locations = df_peak_locations[zone_index]
        return paired_df_peak_locations

    def get_target_zone_bounds(self, anchor_f, anchor_t, f_max, t_max, zone_f_size, zone_t_offset, zone_t_size):
        zone_time_start = anchor_t + zone_t_offset
        zone_time_end = min(t_max, zone_time_start + zone_t_size)
        zone_freq_start = max(0, anchor_f - (zone_f_size // 2))
        zone_freq_end = min(f_max, zone_freq_start + zone_f_size)
        if zone_freq_end == f_max:
            zone_freq_start = zone_freq_end - zone_f_size
        return zone_freq_start, zone_freq_end, zone_time_start, zone_time_end

    def combine_parts_into_key(self, peak_f, second_peak_f, time_delta):
        peak_f = np.uint32(peak_f)
        second_peak_f = np.uint32(second_peak_f)
        time_delta = np.uint32(time_delta)

        first_part = np.left_shift(peak_f, np.uint32(20))
        second_part = np.left_shift(second_peak_f, np.uint32(10))
        combined_key = first_part + second_part + time_delta
        return combined_key

    def decode_hash(self, key):
        # only keep the 10 least significant bits
        time_delta = np.bitwise_and(key, np.uint32(1023))
        # shift 10 bits and only keep the 10 least significant bits
        second_peak_f = np.bitwise_and(np.right_shift(key, np.uint32(10)), np.uint32(1023))
        # shift 20 bits
        peak_f = np.right_shift(key, np.uint32(20))
        return peak_f, second_peak_f, time_delta


def get_mp3_filepaths_from_directory(directory='C:/Users\Luke\Downloads/Disasterpeace/'):
    mp3_filepaths = []
    for filepath in os.listdir(directory):
        if filepath[-4:] != '.mp3':
            continue
        mp3_filepaths.append(directory + filepath)
    return mp3_filepaths


def main(insert_into_database=False):
    audio_prints_db = MongoAudioPrintDB()
    audio_search = AudioSearch(audio_prints_db=audio_prints_db)
    mp3_filepaths = get_mp3_filepaths_from_directory()
    if insert_into_database:
        audio_search.insert_mp3s_fingerprints_into_database(mp3_filepaths)
    else:
        audio_search.measure_performance_of_multiple_snrs_and_mp3s(mp3_filepaths)
    return


if __name__ == '__main__':
    main()
