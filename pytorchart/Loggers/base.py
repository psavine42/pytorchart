import pickle


class BaseLogger(object):
    def __init__(self):
        pass

    def save(self, path):
        pickle.dump(self, path)

    def log(self, *args):
        pass


