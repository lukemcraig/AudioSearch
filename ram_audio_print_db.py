from audio_search_dbs import AudioPrintsDB, DuplicateKeyError
import json
import sys


class RamAudioPrintDB(AudioPrintsDB):
    def __init__(self):
        self.fingerprints_hashtable = self.load_fingerprint_table()
        self.songs_hashtable, self.songtitles_hashtable = self.load_song_table()
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
        songtitles_hashtable = {}
        with open('mongoexport/audioprintsDB.songs.json', mode='r') as f:
            for line in f:
                song = json.loads(line)
                song_id = song.pop("_id")
                song_tuple = (song['artist'], song['album'], song['title'], song['track_length_s'])
                try:
                    songtitles_hashtable[song['title']].append(song_id)
                except KeyError:
                    songtitles_hashtable[song['title']] = [song_id]
                songs_hashtable[song_id] = song_tuple
        print(len(songs_hashtable), "songs")
        return songs_hashtable, songtitles_hashtable

    def insert_one_fingerprint(self, fingerprint):
        return

    def find_one_song(self, song):
        matching_titles = self.songtitles_hashtable[song['title']]
        for possible_song_id in matching_titles:
            possible_song = self.songs_hashtable[possible_song_id]
            if song['artist'] == possible_song[0]:
                if song['album'] == possible_song[1]:
                    if song['track_length_s'] == possible_song[3]:
                        song_doc = {"_id": 0}
                        return song_doc
        return None

    def get_next_song_id(self):
        return

    def insert_one_song(self, song):
        return

    def find_db_fingerprints_with_hash_key(self, fingerprint):
        return
