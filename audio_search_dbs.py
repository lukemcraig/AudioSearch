from abc import ABC, abstractmethod


class DuplicateKeyError(Exception):
    pass


class AudioPrintsDB(ABC):

    @abstractmethod
    def insert_one_fingerprint(self, fingerprint):
        pass

    def find_one_song(self, song):
        pass

    def get_next_song_id(self):
        pass

    def insert_one_song(self, song):
        pass

    def find_db_fingerprints_with_hash_key(self, fingerprint):
        pass
