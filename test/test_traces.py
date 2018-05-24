import sys
# sys.path.append("..")
import pkgutil
from unittest import TestCase
search_path = '.' # set to None to see all modules importable from sys.path
all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
print(all_modules)
from pytorchart import FlexLogger
import random


_nan = float('NaN')


def rand_or_nil(d=None, p=0.5):
    return d if random.random() > p else random.random()


class TestTrace(TestCase):
    def exersize_1(self, Stat, d=None, p=0.5, s=50):
        for i in range(s):
            Stat(train_loss=rand_or_nil(d, p), train_acc=rand_or_nil(d, p))
            if i % 2 == 0:
                Stat(test_loss=rand_or_nil(d, p), test_acc=rand_or_nil(d, p))
            if i % 5 == 0:
                Stat.log(reset=True, step=False)
            Stat.step()

    def exersize_2(self, Stat, d=None, p=0.5, s=50):
        for i in range(s):
            Stat(train_loss=rand_or_nil(d, p), train_acc=rand_or_nil(d, p))
            if i % 2 == 0:
                Stat(test_loss=rand_or_nil(d, p), test_acc=rand_or_nil(d, p))
                Stat.log(reset=True, step=False)
            Stat.step()

    def test_nans(self):
        Stat = FlexLogger.from_presets('loss', 'acc')
        self.exersize_1(Stat, _nan, 0.5)

    def test_nones(self):
        Stat = FlexLogger.from_presets('loss', 'acc')
        self.exersize_1(Stat, d=None, p=0.5)
        self.exersize_2(Stat, d=None, p=0.5)

    def test_nonesH(self):
        Stat = FlexLogger.from_presets('loss', 'acc')
        self.exersize_1(Stat, d=None, p=0.2)
        self.exersize_2(Stat, d=None, p=0.2)
        self.exersize_1(Stat, d=None, p=0.2, s=450)

        print(str(Stat))

    def test_good(self):
        Stat = FlexLogger.from_presets('loss', 'acc')
        self.exersize_1(Stat, None, 1)

