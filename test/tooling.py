import torch
import torch.nn as nn
import pprint, random
import unittest
from src.tooled import TooledModel
from src.modellogger import TooledModelLogger
from torch.autograd import Variable
import torch.nn.functional as F


class TestTools(unittest.TestCase):
    def test_single(self):
        model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))

        TM = TooledModel(model)

        TM.table()
        for i in range(4):
            x = Variable(torch.rand(2, 20))
            targets = Variable(torch.rand(2, 3))
            o = model(x)
            loss = F.mse_loss(o, targets)
            loss.backward()
        TM.table()

    def test_model_log(self):

        model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
        optim = torch.optim.Adam(model.parameters())

        TM = TooledModelLogger(model)
        for i in range(4):
            x = Variable(torch.rand(2, 20))
            targets = Variable(torch.rand(2, 3))

            o = model(x)
            loss = F.mse_loss(o, targets)

            loss.backward()
            TM.step(log=True)
            optim.step()
        TM.table()
        # pprint.pprint(TM.get_dict())



