import pymongo

from audio_search_dbs import AudioPrintsDB, DuplicateKeyError


class MongoAudioPrintDB(AudioPrintsDB):
    def __init__(self):
        self.client = self.get_client()
        self.fingerprints_collection = self.client.audioprintsDB.fingerprints
        self.songs_collection = self.client.audioprintsDB.songs
        pass

    def get_db_fingerprint_song_id(self, fingerprint):
        return fingerprint['songID']

    def get_db_fingerprint_offset(self, fingerprint):
        return fingerprint['offset']

    def insert_one_fingerprint(self, fingerprint):
        try:
            self.fingerprints_collection.insert_one(fingerprint)
        except pymongo.errors.DuplicateKeyError:
            raise DuplicateKeyError
        return

    def insert_many_fingerprints(self, fingerprints):
        self.fingerprints_collection.insert_many(fingerprints)
        return

    def find_one_song(self, song):
        return self.songs_collection.find_one(song)

    def get_next_song_id(self):
        most_recent_song = self.songs_collection.find_one({}, sort=[(u"_id", -1)])
        if most_recent_song is not None:
            new_id = most_recent_song['_id'] + 1
        else:
            new_id = 0
        return new_id

    def insert_one_song(self, song):
        try:
            insert_song_result = self.songs_collection.insert_one(song)
        except pymongo.errors.DuplicateKeyError:
            # increment the key and try again
            song['_id'] += 1
            return self.insert_one_song(song)
        return insert_song_result.inserted_id

    def find_db_fingerprints_with_hash_key(self, fingerprint):
        return self.fingerprints_collection.find({'hash': fingerprint['hash']}, projection={"_id": 0, "hash": 0})

    def get_client(self):
        print("getting client...")
        client = pymongo.MongoClient('mongodb://localhost:27017')
        print("got client", flush=True)
        return client
