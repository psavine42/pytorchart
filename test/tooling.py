import torch
import torch.nn as nn
import pprint, random
import unittest
from src.tooled import TooledModel
from torch.autograd import Variable
import torch.nn.functional as F


class TestTools(unittest.TestCase):
    def test_single(self):
        model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))

        TM = TooledModel(model)

        x = Variable(torch.rand(2, 20))
        targets = Variable(torch.rand(2, 3))

        o = model(x)
        loss = F.mse_loss(o, targets)
        loss.backward()

        TM.table()




