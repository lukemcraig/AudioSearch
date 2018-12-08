from audio_search_dbs import AudioPrintsDB, DuplicateKeyError
import json
import csv
import pickle
import pandas as pd
import sys
from collections import OrderedDict
import numpy as np


class RamAudioPrintDB(AudioPrintsDB):
    def __init__(self):
        self.fingerprints_hashtable = {}
        # self.load_fingerprint_table_from_json()
        self.load_fingerprint_table_from_csv()
        self.save_fingerprint_table_to_pickle()
        self.songs_hashtable = OrderedDict()
        self.songtitles_hashtable = {}
        # self.load_song_tables_from_json()
        self.load_song_tables_from_csv()
        pass

    def save_fingerprint_table_to_pickle(self):

        return

    def load_fingerprint_table_from_json(self):
        with open('mongoexport/small_audioprintsDB.fingerprints.json', mode='r') as f:
            for line in f:
                fingerprint = json.loads(line)
                del fingerprint["_id"]
                try:
                    self.insert_one_fingerprint(fingerprint)
                except DuplicateKeyError:
                    continue
        print(len(self.fingerprints_hashtable), "unique fingerprint hashes")
        # this is not the actual size
        hashtable_size = sys.getsizeof(self.fingerprints_hashtable)
        print(hashtable_size, "bytes")
        return

    def load_fingerprint_table_from_csv(self):
        df_fingerprints = pd.read_csv('mongoexport/audioprintsDB.fingerprints.csv', index_col=0, encoding='utf-8',
                                      dtype=np.uint32)
        df_fingerprints.sort_index(inplace=True)
        self.fingerprints_hashtable = df_fingerprints
        print()
        # with open('mongoexport/audioprintsDB.fingerprints.csv', encoding="utf8", mode='r') as f:
        #     csv_reader = csv.DictReader(f)
        #     for fingerprint in csv_reader:
        #         line_num = csv_reader.line_num
        #         # if line_num > 10000:
        #         #     break
        #         print(line_num)
        #         fingerprint['hash'] = np.uint32(fingerprint['hash'])
        #         fingerprint['offset'] = np.uint16(fingerprint['offset'])
        #         fingerprint['songID'] = np.uint16(fingerprint['songID'])
        #         try:
        #             self.insert_one_fingerprint(fingerprint)
        #         except DuplicateKeyError:
        #             continue
        # print(len(self.fingerprints_hashtable), "unique fingerprint hashes")
        # # this is not the actual size
        # hashtable_size = sys.getsizeof(self.fingerprints_hashtable)
        # print(hashtable_size, "bytes")
        return

    def load_song_tables_from_json(self):
        with open('mongoexport/small_audioprintsDB.songs.json', mode='r') as f:
            for line in f:
                song = json.loads(line)
                self.insert_one_song(song)
        print(len(self.songs_hashtable), "songs")
        return

    def load_song_tables_from_csv(self):
        with open('mongoexport/audioprintsDB.songs.csv', encoding="utf8", mode='r') as f:
            csv_reader = csv.DictReader(f)
            for song in csv_reader:
                song['_id'] = np.uint16(song['_id'])
                song['track_length_s'] = float(song['track_length_s'])
                self.insert_one_song(song)
        print(len(self.songs_hashtable), "songs")
        return

    def insert_one_fingerprint(self, fingerprint):
        hash_ = fingerprint.pop("hash")
        try:
            existing_fingerprints = self.fingerprints_hashtable[hash_]
            for ef in existing_fingerprints:
                if ef == fingerprint:
                    raise DuplicateKeyError
            existing_fingerprints.append(fingerprint)
        except KeyError:
            self.fingerprints_hashtable[hash_] = [fingerprint]
        return

    #     TODO saving the db to disk

    def insert_many_fingerprints(self, fingerprints):
        raise NotImplementedError
        return

    def find_one_song(self, song):
        try:
            matching_titles = self.songtitles_hashtable[song['title']]
            for possible_song_id in matching_titles:
                possible_song = self.songs_hashtable[possible_song_id]
                if song['artist'] == possible_song[0]:
                    if song['album'] == possible_song[1]:
                        # TODO handle no track length
                        if song['track_length_s'] == possible_song[3]:
                            song['_id'] = possible_song_id
                            return song
        except KeyError:
            return None
        return None

    def get_next_song_id(self):
        try:
            most_recent_song_id = next(reversed(self.songs_hashtable))
            return most_recent_song_id + 1
        except StopIteration:
            return 0

    def insert_one_song(self, song):
        song_id = song.pop("_id")
        song_tuple = (song['artist'], song['album'], song['title'], song['track_length_s'])
        try:
            self.songtitles_hashtable[song['title']].append(song_id)
        except KeyError:
            self.songtitles_hashtable[song['title']] = [song_id]
        self.songs_hashtable[song_id] = song_tuple
        return song_id

    def find_db_fingerprints_with_hash_key(self, fingerprint):
        # self.fingerprints_hashtable.reset_index().reset_index().set_index(["hash", "index"])
        try:
            # TODO
            # position = self.fingerprints_hashtable.index.searchsorted(fingerprint['hash'])
            return self.fingerprints_hashtable.loc[fingerprint['hash']]
        except KeyError:
            return None
