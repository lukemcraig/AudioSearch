import json
import os
import sys
import threading
import time
import timeit
import random
from multiprocessing import Process
from queue import Queue

import numpy as np
import scipy.signal
import scipy.ndimage.filters
# import scipy.ndimage.measurements
import pandas as pd

# import bintrees.rbtree

import librosa
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError

from audio_search_dbs import DuplicateKeyError
# TODO conditional imports
from mongo_audio_print_db import MongoAudioPrintDB
from ram_audio_print_db import RamAudioPrintDB

from audio_search_plotting import plot_recognition_rate, plot_spectrogram_and_peak_subplots_detailed, start_hist_subplots, \
    make_next_hist_subplot, show_hist_plot, plot_hist_of_stks, plot_show, plot_scatter_of_fingerprint_offsets, \
    plot_spectrogram_peaks, plot_spectrogram_and_peak_subplots, finish_scatter_of_fingerprint_offsets, use_ggplot, \
    plot_target_zone, reset_plot_lims, plot_spectrogram


class AudioSearch:
    time_functions = False
    time_add_noise = False & time_functions
    time_find_spec_peaks = False & time_functions
    time_get_target_zone_bounds = False & time_functions
    time_query_peaks_for_target_zone = False & time_functions
    time_query_peaks_for_target_zone_bs = False & time_functions
    time_get_df_of_fingerprint_offsets = False & time_functions

    time_n_repeats = 100

    def __init__(self, audio_prints_db, do_plotting=False, noise_type='White'):
        self.audio_prints_db = audio_prints_db
        self.do_plotting = do_plotting
        self.noise_type = noise_type
        self.pub_data = None

    def insert_mp3s_fingerprints_into_database(self, mp3_filepaths, skip_existing_songs=False):
        for mp3_i, mp3_filepath in enumerate(mp3_filepaths):
            try:
                mp3_metadata = get_mp3_metadata(mp3_filepath)
            except KeyError:
                # this song doesn't have the required metadata, so we'll just skip it
                continue
            except ID3NoHeaderError:
                # this song doesn't have the required metadata, so we'll just skip it
                continue

            if skip_existing_songs:
                # loading the audio data is slow so we optionally skip already added ones, without checking track length
                _, song_doc = self.get_song_from_db_with_metadata_except_length(mp3_metadata)
                if song_doc is not None:
                    continue
            try:
                print(mp3_filepath, flush=True)
            except UnicodeEncodeError:
                print(mp3_filepath.encode('ascii', 'ignore'), flush=True)
            data, rate, metadata = load_audio_data_and_meta(mp3_filepath)
            fingerprints = self.get_fingerprints_from_audio(data, rate)
            sys.stdout.flush()
            self.insert_one_mp3_with_fingerprints_into_database(metadata, fingerprints)
            sys.stdout.flush()
        return

    def measure_performance_of_multiple_snrs_and_mp3s(self, usable_mp3s):
        snrs_to_test = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
        # snrs_to_test = [300]
        print("testing", usable_mp3s, "at", snrs_to_test, "dBs each")
        subset_clip_lengths = [15, 10, 5]
        if self.do_plotting or True:
            markers = ["D", "s", "^"]
            linestyles = ['-', '--', ':']
        performance_results_list = [np.zeros((len(usable_mp3s), len(snrs_to_test)), dtype=bool) for _ in
                                    range(len(subset_clip_lengths))]

        audio_queue = Queue()
        producer = threading.Thread(
            target=load_audio_data_into_queue,
            args=(audio_queue, usable_mp3s),
            name='producer',
        )
        producer.setDaemon(True)
        producer.start()
        # load_audio_data_into_queue(audio_queue, usable_mp3s)
        start_time = time.time()
        for mp3_i, mp3_filepath in enumerate(usable_mp3s):
            print("mp3_i", mp3_i)
            print("queue size:", audio_queue.qsize())
            data, rate, metadata = audio_queue.get()  # load_audio_data(mp3_filepath)

            print(mp3_i, mp3_filepath, "/", len(usable_mp3s))
            for clip_len_i, subset_clip_length in enumerate(subset_clip_lengths):
                print("subset_clip_length:", subset_clip_length, "sec")
                performance_results = performance_results_list[clip_len_i]
                data_subset = self.get_test_subset(data, subset_length=subset_clip_length * rate)
                for snr_i, snr_db in enumerate(snrs_to_test):
                    correct_match, predicted_song_id = self.add_noise_and_predict_one_clip(data_subset, metadata,
                                                                                           mp3_filepath, rate, snr_db)
                    print("snr:", snr_db, ", correct_match:", correct_match)
                    performance_results[mp3_i, snr_i] = correct_match
        end_time = time.time()
        print("elapsed wall time=", end_time - start_time, "seconds")
        if len(usable_mp3s) > 0:
            for clip_len_i, subset_clip_length in enumerate(subset_clip_lengths):
                performance_results = performance_results_list[clip_len_i]
                np.savetxt("perf_results\\performance_results_%d.csv" % subset_clip_length, performance_results,
                           delimiter=',', header=str(snrs_to_test))
                recognition_rate = performance_results.mean(axis=0) * 100.0
                if self.do_plotting or True:
                    plot_recognition_rate(recognition_rate, snrs_to_test, len(usable_mp3s),
                                          clips_length=subset_clip_length, marker=markers[clip_len_i],
                                          linestyle=linestyles[clip_len_i], noise_type=self.noise_type)
        if self.do_plotting or True:
            plot_show()

        return

    def add_noise_and_predict_one_clip(self, data_subset, metadata, mp3_filepath, rate, snr_db):
        data_and_noise = self.add_noise(data_subset, desired_snr_db=snr_db)
        if self.time_add_noise:
            avg_time_add_noise = self.time_a_function(lambda: self.add_noise(data_subset, desired_snr_db=snr_db))
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
        avg_peaks_per_second = len(peak_locations) / t[-1]
        # print('avg_peaks_per_second', avg_peaks_per_second)
        if self.time_find_spec_peaks:
            avg_time = self.time_a_function(lambda: self.find_spectrogram_peaks(Sxx, t_step))
            print("Sxx was ", Sxx.shape)
            print("find_spectrogram_peaks() took", '{0:.2f}'.format(avg_time * 1000), "ms")
        if self.do_plotting:
            # plot_spectrogram(Sxx)
            # plot_show()
            plot_spectrogram_and_peak_subplots_detailed(Sxx, f, max_filter, max_filter_size, peak_locations, t)

        fingerprints = self.get_fingerprints_from_peaks(len(f) - 1, f_step, peak_locations, len(t) - 1, t_step)
        return fingerprints

    def try_to_match_clip_to_database(self, filepath, fingerprints, metadata):
        # print("querying song in database")
        _, song_doc = self.get_song_from_db_with_metadata(metadata)
        if song_doc is None:
            raise Exception(filepath + "needs to be inserted into the DB first!")
        # print("querying database")
        df_fingerprint_matches = self.get_df_of_fingerprint_offsets(fingerprints)

        if self.time_get_df_of_fingerprint_offsets:
            avg_time = self.time_a_function(lambda: self.get_df_of_fingerprint_offsets(fingerprints))
            print("get_df_of_fingerprint_offsets() took", '{0:.2f}'.format(avg_time * 1000), "ms")

        index_set = set(df_fingerprint_matches.index)
        n_possible_songs = len(index_set)
        if n_possible_songs == 0:
            # there were no fingerprints found, so we return an incorrect match result
            return -1, False

        max_hist_song = self.get_the_most_likely_song_from_all_the_histograms(df_fingerprint_matches, n_possible_songs,
                                                                              index_set)
        # TODO false positives?
        correct_match = max_hist_song == song_doc['_id']
        # print("correct_match=", correct_match)
        if self.do_plotting:
            show_hist_plot(max_hist_song, song_doc)
        return max_hist_song, correct_match

    def get_the_most_likely_song_from_all_the_histograms(self, df_fingerprint_matches, n_possible_songs, index_set):
        print("n_possible_songs", n_possible_songs)
        # unique_stks, unique_stks_counts = np.unique(df_fingerprint_matches, return_counts=True)
        # stks_sorted_by_frequency = unique_stks[np.argsort(unique_stks_counts)][::-1]
        # df_fingerprint_matches[df_fingerprint_matches['stk'] == stks_sorted_by_frequency[0]]

        # unique_songs, unique_songs_counts = np.unique(df_fingerprint_matches.index, return_counts=True)
        # songs_sorted_by_frequency = unique_songs[np.argsort(unique_songs_counts)][::-1]

        # df_fingerprint_matches = df_fingerprint_matches.loc[songs_sorted_by_frequency]
        if self.do_plotting:
            # we don't want 4000 subplots
            n_subplots = min(n_possible_songs, 2)
            ax = start_hist_subplots(n_subplots)
        max_hist_peak = 0
        max_hist_song = None
        # for i, song_id in enumerate([2829, 5893, 9496]):
        for i, song_id in enumerate(index_set):
            # print(i)
            stks_in_songID = df_fingerprint_matches.loc[song_id]
            if self.do_plotting:
                if i < n_subplots:
                    make_next_hist_subplot(ax, i, n_subplots, song_id, len(stks_in_songID))
            # make a histogram with bin width of 1
            unique, unique_counts = np.unique(stks_in_songID.values, return_counts=True)
            unique_max = unique.max()
            unique_min = unique.min()
            hist = np.zeros(1 + unique_max - unique_min)
            hist[unique - unique_min] = unique_counts
            # smooth histogram for the sake of "clustered peak detection"
            filtered_hist = scipy.ndimage.filters.uniform_filter1d(hist, size=2, mode='constant')
            # filtered_hist = hist
            max_filtered_hist = filtered_hist.max()
            if max_filtered_hist > max_hist_peak:
                max_hist_peak = max_filtered_hist
                max_hist_song = song_id
            if self.do_plotting:
                if i < n_subplots:
                    plot_hist_of_stks(np.arange(unique_min, unique_max + 1), hist)
                    # overlay the filtered histogram
                    # plot_hist_of_stks(np.arange(unique_min, unique_max + 1), filtered_hist, alpha=0.5)
        return max_hist_song

    def get_df_of_fingerprint_offsets(self, fingerprints):
        stks = []
        db_fp_song_ids = []
        db_fp_offsets = []
        local_fp_offsets = []
        for fingerprint_i, fingerprint in enumerate(fingerprints):
            # print(fingerprint_i)
            db_fp_iterator = self.audio_prints_db.find_db_fingerprints_with_hash_key(fingerprint)
            if db_fp_iterator is not None:
                for db_fp in db_fp_iterator:
                    db_fp_song_id = self.audio_prints_db.get_db_fingerprint_song_id(db_fp)
                    db_fp_song_ids.append(db_fp_song_id)
                    # print(db_fp_song_id)
                    db_fp_offset = self.audio_prints_db.get_db_fingerprint_offset(db_fp)
                    db_fp_offsets.append(db_fp_offset)

                    local_fp_offset = fingerprint['offset']
                    local_fp_offsets.append(local_fp_offset)

                    if self.do_plotting:
                        # if db_fp_song_id == 1062:
                        # if db_fp_song_id == 6078:
                        plot_scatter_of_fingerprint_offsets(fingerprint_i, db_fp_offset, db_fp_song_id,
                                                            local_fp_offset,
                                                            len(fingerprints))

                    stk = db_fp_offset - local_fp_offset
                    stks.append(stk)
        if self.do_plotting:
            finish_scatter_of_fingerprint_offsets()
            plot_show()
        df_fingerprint_matches = pd.DataFrame({
            "songID": db_fp_song_ids,
            "stk": stks
        })
        df_fingerprint_matches.set_index('songID', inplace=True)
        return df_fingerprint_matches

    def insert_one_mp3_with_fingerprints_into_database(self, metadata, fingerprints):
        song_id_in_db = self.get_or_insert_song_into_db(metadata)
        print("inserting fingerprints into database, songID=" + str(song_id_in_db), flush=True)
        self.insert_list_of_fingerprints(fingerprints, song_id_in_db)
        return

    def insert_list_of_fingerprints(self, fingerprints, song_id_in_db):
        for fingerprint in fingerprints:
            fingerprint['songID'] = song_id_in_db
        self.audio_prints_db.insert_many_fingerprints(fingerprints)
        print("finished fingerprints into database, songID=" + str(song_id_in_db), flush=True)
        return

    def get_or_insert_song_into_db(self, metadata):
        print("querying song in database")
        song, song_doc = self.get_song_from_db_with_metadata(metadata)
        if song_doc is None:
            print("inserting song into database")
            new_id = self.audio_prints_db.get_next_song_id()
            if new_id > 10000:
                raise Exception("We reached 10,000 songs, don't insert any more.")
            song['_id'] = new_id

            inserted_id = self.audio_prints_db.insert_one_song(song)
            print("songID=", inserted_id)
            return inserted_id
        else:
            return song_doc['_id']

    def get_song_from_db_with_metadata(self, metadata):
        song = {'artist': metadata['artist'], 'album': metadata['album'], 'title': metadata['title'],
                'track_length_s': metadata['track_length_s']}
        song_doc = self.audio_prints_db.find_one_song(song)
        return song, song_doc

    def get_song_from_db_with_metadata_except_length(self, metadata):
        song = {'artist': metadata['artist'], 'album': metadata['album'], 'title': metadata['title']}
        song_doc = self.audio_prints_db.find_one_song(song)
        return song, song_doc

    def get_test_subset(self, data, subset_length):
        # subset_length = np.random.randint(rate * 5, rate * 14)
        # subset_length = int(8000 * 15)
        subset_length = min(len(data), subset_length)
        # random = np.random.RandomState(42)
        # random_start_time = random.randint(0, len(data) - subset_length)

        # test from the middle
        start_time = (len(data) // 2) - (subset_length // 2)
        start_time = max(start_time, 0)
        data = data[start_time:start_time + subset_length]
        return data

    def add_noise(self, data, desired_snr_db):
        if self.noise_type == 'Pub':
            noise = self.get_pub_noise(data)
        else:
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

    def convert_to_db(self, linear_values):
        return 20 * np.log10(linear_values)

    def get_rms_linear(self, data):
        return np.sqrt(np.mean(np.square(data)))

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
        peak_locations = np.argwhere((Sxx == max_filter) & (Sxx != 0))
        return peak_locations, max_filter, (t_size, f_size)

    def get_fingerprints_from_peaks(self, f_max, f_step, peak_locations, t_max, t_step):
        # print("get_fingerprints_from_peaks")
        n_peaks = len(peak_locations)
        print("n_peaks=", n_peaks)
        # TODO fan out factor
        fan_out_factor = 10
        # 1400hz tall zone box
        zone_f_size = 1400 // f_step
        # 6 second wide zone box
        zone_t_size = 6 // t_step
        # start one spectrogram time segment after the current one
        zone_t_offset = 1
        df_peak_locations = pd.DataFrame(peak_locations, columns=['f', 't'])

        # sort by time
        df_peak_locations.sort_values(by='t', ascending=True, inplace=True)
        peak_locations_t_sort = df_peak_locations['t']
        # sort by frequency
        peak_locations_f_sort = df_peak_locations['f'].sort_values(ascending=True)

        # sorted_t_location = df_peak_locations.values.__array_interface__['data'][0]
        # sorted_f_location = df_peak_locations_f_sort.values.__array_interface__['data'][0]
        fingerprints = []
        avg_n_pairs_per_peak = 0

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

            if self.time_get_target_zone_bounds:
                avg_time = self.time_a_function(
                    lambda: self.get_target_zone_bounds(anchor_f, anchor_t, f_max, t_max, zone_f_size,
                                                        zone_t_offset, zone_t_size))
                print("get_target_zone_bounds() took", '{0:.2f}'.format(avg_time * 1000), "ms")

            # paired_df_peak_locations_sweep, n_pairs_sweep = self.query_dataframe_for_peaks_in_target_zone_sweep_lines(
            #     df_peak_locations, peak_locations_t_sort, peak_locations_f_sort,
            #     zone_freq_end, zone_freq_start, zone_time_end, zone_time_start)

            # TODO better way to check the zone (sweep line)
            paired_df_peak_locations, n_pairs = self.query_dataframe_for_peaks_in_target_zone_binary_search(
                df_peak_locations, peak_locations_t_sort, peak_locations_f_sort,
                zone_freq_end, zone_freq_start, zone_time_end, zone_time_start)
            if self.time_query_peaks_for_target_zone_bs:
                avg_time = self.time_a_function(
                    lambda: self.query_dataframe_for_peaks_in_target_zone_binary_search(
                        df_peak_locations, peak_locations_t_sort, peak_locations_f_sort,
                        zone_freq_end, zone_freq_start, zone_time_end,
                        zone_time_start))
                print("query_dataframe_for_peaks_in_target_zone_binary_search() took",
                      '{0:.2f}'.format(avg_time * 1000), "ms")

            old_peaks_in_target_zone_method = False
            if old_peaks_in_target_zone_method:
                paired_df_peak_locations_old, n_pairs_old = self.query_dataframe_for_peaks_in_target_zone(
                    df_peak_locations, zone_freq_end, zone_freq_start, zone_time_end, zone_time_start)
                assert n_pairs == n_pairs_old
                pd.testing.assert_frame_equal(paired_df_peak_locations, paired_df_peak_locations_old)
                if self.time_query_peaks_for_target_zone:
                    avg_time = self.time_a_function(
                        lambda: self.query_dataframe_for_peaks_in_target_zone(df_peak_locations,
                                                                              zone_freq_end,
                                                                              zone_freq_start,
                                                                              zone_time_end,
                                                                              zone_time_start))
                    print("query_dataframe_for_peaks_in_target_zone() took", '{0:.2f}'.format(avg_time * 1000), "ms")

            avg_n_pairs_per_peak += n_pairs

            for j, second_peak in paired_df_peak_locations.iterrows():
                # print("    ", j, "/", n_pairs)
                second_peak_f = second_peak['f']
                second_peak_t_ = second_peak['t']
                time_delta = second_peak_t_ - anchor_t
                combined_key = self.combine_parts_into_key(anchor_f, second_peak_f, time_delta)
                # print(combined_key)
                fingerprint = {'hash': int(combined_key), 'offset': int(anchor_t)}
                fingerprints.append(fingerprint)

            if self.do_plotting:
                print(i, anchor_t, anchor_f)
                use_ggplot()
                reset_plot_lims()
                plot_spectrogram_peaks(peak_locations)
                plot_target_zone(zone_freq_start, zone_freq_end, zone_time_start, zone_time_end, anchor_t, anchor_f,
                                 second_peak_t_, second_peak_f)
                plot_show()
        # df_fingerprints = pd.DataFrame(fingerprints)
        avg_n_pairs_per_peak /= n_peaks
        # print("avg_n_pairs_per_peak", avg_n_pairs_per_peak)
        return fingerprints

    def query_dataframe_for_peaks_in_target_zone_sweep_lines(self, df_peak_locations, peak_locations_t,
                                                             peak_locations_f,
                                                             zone_freq_end, zone_freq_start,
                                                             zone_time_end, zone_time_start):
        start = peak_locations_t.searchsorted(zone_time_start, side='left')[0]
        end = peak_locations_t.searchsorted(zone_time_end, side='right')[0]
        t_index = peak_locations_t.index[start:end]

        f_start = peak_locations_f.searchsorted(zone_freq_start, side='left')[0]
        f_end = peak_locations_f.searchsorted(zone_freq_end, side='right')[0]
        f_index = peak_locations_f.index[f_start:f_end]

        paired_df_peak_locations = df_peak_locations.loc[t_index & f_index]

        n_pairs = len(paired_df_peak_locations)

        return paired_df_peak_locations, n_pairs

    def query_dataframe_for_peaks_in_target_zone_binary_search(self, df_peak_locations, peak_locations_t,
                                                               peak_locations_f,
                                                               zone_freq_end, zone_freq_start,
                                                               zone_time_end, zone_time_start):
        start = peak_locations_t.searchsorted(zone_time_start, side='left')[0]
        end = peak_locations_t.searchsorted(zone_time_end, side='right')[0]
        t_index = peak_locations_t.index[start:end]

        f_start = peak_locations_f.searchsorted(zone_freq_start, side='left')[0]
        f_end = peak_locations_f.searchsorted(zone_freq_end, side='right')[0]
        f_index = peak_locations_f.index[f_start:f_end]

        paired_df_peak_locations = df_peak_locations.loc[t_index & f_index]

        n_pairs = len(paired_df_peak_locations)

        return paired_df_peak_locations, n_pairs

    def query_dataframe_for_peaks_in_target_zone(self, df_peak_locations, zone_freq_end, zone_freq_start, zone_time_end,
                                                 zone_time_start):
        # these are all actually boolean dataframes, not indexes
        time_index = (df_peak_locations['t'] <= zone_time_end) & (df_peak_locations['t'] >= zone_time_start)
        freq_index = (zone_freq_start <= df_peak_locations['f']) & (df_peak_locations['f'] <= zone_freq_end)
        zone_index = time_index & freq_index
        n_pairs = zone_index.sum()
        # print("n_pairs:", n_pairs)
        paired_df_peak_locations = df_peak_locations[zone_index]
        return paired_df_peak_locations, n_pairs

    def get_target_zone_bounds(self, anchor_f, anchor_t, f_max, t_max, zone_f_size, zone_t_offset, zone_t_size):
        zone_time_start = anchor_t + zone_t_offset
        zone_time_end = min(t_max, zone_time_start + zone_t_size)
        zone_freq_start = max(0, anchor_f - (zone_f_size // 2))
        zone_freq_end = min(f_max, zone_freq_start + zone_f_size)
        if zone_freq_end == f_max:
            zone_freq_start = zone_freq_end - zone_f_size
        return int(zone_freq_start), int(zone_freq_end), int(zone_time_start), int(zone_time_end)

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

    def get_pub_noise(self, data):
        # cache it
        if self.pub_data is None:
            pub_data, rate = load_audio_data('noise_sample\\pub.wav')
            self.pub_data = pub_data
        return self.pub_data[:len(data)]


def get_mp3_metadata(filepath):
    mp3tags = EasyID3(filepath)
    metadata = {
        "artist": mp3tags['artist'][0],
        "album": mp3tags['album'][0],
        "title": mp3tags['title'][0]
    }
    return metadata


def load_audio_data_and_meta(filepath):
    # print("loading audio", flush=True)
    data, rate = load_audio_data(filepath)
    metadata = get_mp3_metadata(filepath)
    metadata["track_length_s"] = len(data) / rate
    return data, rate, metadata


def load_audio_data(filepath):
    desired_rate = 8000
    data, rate = librosa.load(filepath, mono=True, sr=desired_rate)
    assert rate == desired_rate
    return data, rate


def get_mp3_genres(filepath):
    mp3tags = EasyID3(filepath)
    try:
        genres = mp3tags['genre']
    except KeyError:
        genres = ['Unknown']
    return genres


def get_mp3_filepaths_from_directory(
        directory='G:\\Users\\Luke\\Music\\iTunes\\iTunes Media\\Music\\A Tribe Called Quest\\Midnight Marauders\\'):
    mp3_filepaths = []
    for filepath in os.listdir(directory):
        if filepath[-4:] != '.mp3':
            continue
        mp3_filepaths.append(directory + filepath)
    return mp3_filepaths


def get_n_random_mp3s_to_test(audio_search, root_directory, test_size):
    mp3_filepaths_to_test = []
    for directory, subdirs, file_names in os.walk(root_directory):
        mp3_filepaths = [os.path.join(directory, fp) for fp in file_names if fp.endswith('.mp3')]
        if len(mp3_filepaths) > 0:
            for mp3_i, mp3_filepath in enumerate(mp3_filepaths[0:1]):
                try:
                    mp3_metadata = get_mp3_metadata(mp3_filepath)
                except KeyError:
                    # this song doesn't have the required metadata, so we'll just skip it
                    continue
                _, song_doc = audio_search.get_song_from_db_with_metadata_except_length(mp3_metadata)
                if song_doc is None:
                    # This song wasn't already in the database
                    continue
                mp3_filepaths_to_test.append(mp3_filepath)
        #         if len(mp3_filepaths_to_test) >= test_size:
        #             break
        # if len(mp3_filepaths_to_test) >= test_size:
        #     break
    mp3_filepaths_to_test = random.sample(mp3_filepaths_to_test, test_size)
    return mp3_filepaths_to_test


def load_audio_data_into_queue(audio_queue, usable_mp3s):
    for mp3_i, mp3_filepath in enumerate(usable_mp3s):
        # print(mp3_i, mp3_filepath, "/", len(usable_mp3s))
        data, rate, metadata = load_audio_data_and_meta(mp3_filepath)
        audio_queue.put((data, rate, metadata))
    return


def get_test_set_and_test(audio_search, root_directory):
    # test_list_json_read_path = None
    test_list_json_read_path = 'song_test_sets\\test_mp3_paths_.json'
    if test_list_json_read_path is not None:
        with open(test_list_json_read_path, 'r')as json_fp:
            mp3_filepaths_to_test = json.load(json_fp)
    else:
        test_size = 3
        mp3_filepaths_to_test = get_n_random_mp3s_to_test(audio_search, root_directory, test_size)
        test_list_json_write_path = 'song_test_sets\\test_mp3_paths_.json'
        with open(test_list_json_write_path, 'w')as json_fp:
            json.dump(mp3_filepaths_to_test, json_fp)

    # TODO plot genre counts?
    unique_genres, unique_genres_counts = get_distribution_of_genres(mp3_filepaths_to_test)
    print(unique_genres)
    print(unique_genres_counts)

    audio_search.measure_performance_of_multiple_snrs_and_mp3s(mp3_filepaths_to_test)
    return


def get_distribution_of_genres(mp3_filepaths_to_test):
    genres = []
    for mp3_path in mp3_filepaths_to_test:
        mp3_genres = get_mp3_genres(mp3_path)
        genres += [g for g in mp3_genres if not g.startswith('http')]
    unique_genres, unique_genres_counts = np.unique(genres, return_counts=True)
    return unique_genres, unique_genres_counts


def connect_to_database_and_insert_mp3s_fingerprints_into_database(audio_prints_db, mp3_filepaths):
    sys.stdout = open("logs\\insert_" + str(os.getpid()) + ".log", "a")
    sys.stderr = open("logs\\insert_" + str(os.getpid()) + "_err.log", "a")
    audio_search = AudioSearch(audio_prints_db=audio_prints_db())
    sys.stdout.flush()
    audio_search.insert_mp3s_fingerprints_into_database(mp3_filepaths, skip_existing_songs=True)
    return


def insert_mp3s_from_directory_in_random_order(audio_prints_db, root_directory, n_processes):
    all_mp3_file_paths = []
    for directory, _, file_names in os.walk(root_directory):
        mp3_filepaths = [os.path.join(directory, fp) for fp in file_names if fp.endswith('.mp3')]
        if len(mp3_filepaths) > 0:
            all_mp3_file_paths += mp3_filepaths
    # shuffle the order of insertion so if we don't use all the mp3s we'll get a random sample
    random.shuffle(all_mp3_file_paths)
    process_list = []
    split_mp3_list = np.array_split(all_mp3_file_paths, n_processes)
    for i, all_mp3_file_paths_for_proc in enumerate(split_mp3_list):
        print("spawning process", i, "for, at most,", len(all_mp3_file_paths_for_proc), "mp3s")
        p = Process(target=connect_to_database_and_insert_mp3s_fingerprints_into_database,
                    args=(audio_prints_db, all_mp3_file_paths_for_proc.tolist(),))
        p.start()
        process_list.append(p)
    while True:
        all_finished = True
        for p in process_list:
            p.join(1)
            p_is_alive = p.is_alive()
            print(p.pid, "is alive?:", p_is_alive)
            if p_is_alive:
                all_finished = False
            # else:
            #     process_list.remove(p)
        print("---")
        if all_finished:
            break
        time.sleep(10)

    return


def main(insert_into_database=False, root_directory='G:\\Users\\Luke\\Music\\iTunes\\iTunes Media\\Music\\'):
    audio_prints_db = MongoAudioPrintDB
    # audio_prints_db = RamAudioPrintDB
    if insert_into_database:
        insert_mp3s_from_directory_in_random_order(audio_prints_db, root_directory, n_processes=1)
    else:
        audio_search = AudioSearch(audio_prints_db=audio_prints_db(), do_plotting=False, noise_type='Pub')
        get_test_set_and_test(audio_search, root_directory)
    return


# TODO command line interface

if __name__ == '__main__':
    main()
