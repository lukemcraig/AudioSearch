from audio_search_dbs import AudioPrintsDB, DuplicateKeyError
import json
import sys


class RamAudioPrintDB(AudioPrintsDB):
    def __init__(self):
        fingerprints_hashtable = self.load_fingerprint_table()
        songs_hashtable = self.load_song_table()
        pass

    def load_fingerprint_table(self):
        fingerprint_hashtable = {}
        with open('mongoexport/audioprintsDB.fingerprints.json', mode='r') as f:
            for line in f:
                fingerprint = json.loads(line)
                del fingerprint["_id"]
                hash_ = fingerprint.pop("hash")
                fingerprint_tuple = (fingerprint['offset'], fingerprint['songID'])
                try:
                    fingerprint_hashtable[hash_].append(fingerprint_tuple)
                except KeyError:
                    fingerprint_hashtable[hash_] = [fingerprint_tuple]
        print(len(fingerprint_hashtable), "unique fingerprint hashes")
        # this is not the actual size
        hashtable_size = sys.getsizeof(fingerprint_hashtable)
        print(hashtable_size, "bytes")
        return fingerprint_hashtable

    def load_song_table(self):
        songs_hashtable = {}
        with open('mongoexport/audioprintsDB.songs.json', mode='r') as f:
            for line in f:
                song = json.loads(line)
                song_id = song.pop("_id")
                song_tuple = (song['artist'], song['album'], song['title'], song['track_length_s'])
                try:
                    songs_hashtable[song_id].append(song_tuple)
                except KeyError:
                    songs_hashtable[song_id] = [song_tuple]
        print(len(songs_hashtable), "songs")
        return songs_hashtable

    def insert_one_fingerprint(self, fingerprint):
        return

    def find_one_song(self, song):
        return

    def get_next_song_id(self):
        return

    def insert_one_song(self, song):
        return

    def find_db_fingerprints_with_hash_key(self, fingerprint):
        return
