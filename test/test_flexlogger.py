import torch
import pprint, random
import unittest
from logutils import FlexLogger


def random_update(keys):
    return {k: random.random() for k in keys}


def mse_values(keys):
    size = [4, 20]
    return {k: [torch.randn(*size), torch.randn(*size)] for k in keys}


class TestLoggers(unittest.TestCase):
    def setUp(self):
        self.LG1 = FlexLogger.from_presets('loss', 'mse')
        # pprint.pprint(self.LG1)
        self.step = 0

    def test_single(self):
        for i in range(5):
            self.step += 1
            self.LG1.add(random_update({'test_loss'}))
            self.LG1.add(random_update({'train_loss'}))
            self.LG1.log(self.step)
        self.LG1.log(self.step+1)

    def test_mse(self):
        for i in range(5):
            self.step += 1
            self.LG1.add(random_update({'test_loss', 'train_loss'}))
            self.LG1.add(mse_values({'train_mse'}))
            self.LG1(test_mse=[torch.randn([4, 20]), torch.randn([4, 20])])
            self.LG1.log(self.step)
        print(str(self.LG1))



