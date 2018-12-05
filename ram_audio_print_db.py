from audio_search_dbs import AudioPrintsDB, DuplicateKeyError
import json
import sys
from collections import OrderedDict


class RamAudioPrintDB(AudioPrintsDB):
    def __init__(self):
        self.fingerprints_hashtable = {}
        self.load_fingerprint_table_from_json()

        self.songs_hashtable = OrderedDict()
        self.songtitles_hashtable = {}
        self.load_song_tables_from_json()
        pass

    def load_fingerprint_table_from_json(self):
        with open('mongoexport/audioprintsDB.fingerprints.json', mode='r') as f:
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

    def load_song_tables_from_json(self):
        with open('mongoexport/audioprintsDB.songs.json', mode='r') as f:
            for line in f:
                song = json.loads(line)
                self.insert_one_song(song)
        print(len(self.songs_hashtable), "songs")
        return

    #     TODO saving the db to disk

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

    def find_one_song(self, song):
        try:
            matching_titles = self.songtitles_hashtable[song['title']]
            for possible_song_id in matching_titles:
                possible_song = self.songs_hashtable[possible_song_id]
                if song['artist'] == possible_song[0]:
                    if song['album'] == possible_song[1]:
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
        try:
            return self.fingerprints_hashtable[fingerprint['hash']]
        except KeyError:
            return None
