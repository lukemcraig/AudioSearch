from audio_search_dbs import AudioPrintsDB, DuplicateKeyError
import json
import sys


class RamAudioPrintDB(AudioPrintsDB):
    def __init__(self):
        fingerprint_hashtable = {}
        with open('mongoexport/audioprintsDB.fingerprints.json', mode='r') as f:
            for line in f:
                fingerprint = json.loads(line)
                del fingerprint["_id"]
                hash = fingerprint.pop("hash")
                fingerprint_tuple = (fingerprint['offset'], fingerprint['songID'])
                try:
                    fingerprint_hashtable[hash].append(fingerprint_tuple)
                except KeyError:
                    fingerprint_hashtable[hash] = [fingerprint_tuple]
        print(len(fingerprint_hashtable), "unique fingerprint hashes")
        # this is not the actual size
        hashtable_size = sys.getsizeof(fingerprint_hashtable)
        print(hashtable_size, "bytes")

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
