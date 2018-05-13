import torch
import pprint, random
import unittest
from src.logging import FlexLogger
from src.preconfigured import get_preset_logger


def random_update(keys):
    return {k: random.random() for k in keys}


def mse_values(keys):
    size = [4, 20]
    return {k: [torch.randn(*size), torch.randn(*size)] for k in keys}


class TestLoggers(unittest.TestCase):
    def setUp(self):
        self.LG1 = get_preset_logger('loss+MSE')
        pprint.pprint(self.LG1.get_plot_definitions())
        print(self.LG1._meters)
        print(self.LG1._plots)
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
            self.LG1.add(mse_values({'test_mse'}))
            self.LG1.log(self.step)


