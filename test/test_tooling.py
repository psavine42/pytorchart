import torch
import torch.nn as nn
import pprint, random
import unittest
from logutils import TooledModel,  TooledModelLogger, FlexTooledModel
from logutils.preconfigured import get_preset
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
        model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(),  nn.Linear(10, 3))
        optim = torch.optim.Adam(model.parameters())
        TM = TooledModelLogger(model, metrics=['grad_out', 'weights'], funcs=[torch.mean])

        assert len(TM.get_handles()) == 6
        assert TM._TM._opts == ['grad_out', 'weights']
        # loop
        for i in range(4):
            optim.zero_grad()
            x = Variable(torch.rand(2, 20))
            targets = Variable(torch.rand(2, 3))

            o = model(x)
            loss = F.mse_loss(o, targets)
            TM.step(log=True)
            loss.backward()
            optim.step()
        TM.table()

    def _spec(self, spec):
        model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 3))
        optim = torch.optim.Adam(model.parameters())
        TM = TooledModelLogger(model,  spec=spec)

        for i in range(4):
            optim.zero_grad()
            x = Variable(torch.rand(2, 20))
            targets = Variable(torch.rand(2, 3))

            o = model(x)
            loss = F.mse_loss(o, targets)
            TM.step(log=True)
            loss.backward()
            optim.step()
        TM.table()

    def _spec_plus_add(self, spec, meter_args=None, plot_args=None, mode='by_layer'):
        model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 3))
        optim = torch.optim.Adam(model.parameters())
        TM = TooledModelLogger(
            model, spec=spec, meter_args=meter_args, plot_args=plot_args, mode=mode)

        print(str(TM))
        for i in range(4):
            optim.zero_grad()
            x = Variable(torch.rand(2, 20))
            targets = Variable(torch.rand(2, 3))

            o = model(x)
            loss = F.mse_loss(o, targets)
            loss.backward()
            optim.step()
            print(loss)
            TM.add({'train_loss': loss.data[0], 'train_mse': [o.data, targets.data]})
            TM.log()
        TM.table()

    def _spec_logger2(self, meter_args=None, plot_args=None):
        model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 3))
        optim = torch.optim.Adam(model.parameters())
        print(model)
        TM = FlexTooledModel(plot_args, meter_args, model)
        print(str(TM.show(meta=True)))
        for i in range(4):
            optim.zero_grad()
            x = Variable(torch.rand(2, 20))
            targets = Variable(torch.rand(2, 3))
            o = model(x)
            loss = F.mse_loss(o, targets)
            loss.backward()
            optim.step()
            TM.log()
            TM.step()

    def test_spec_all(self):
        spec = {'': {'weights': [torch.mean, torch.std],
                     'grad_out': [torch.mean, torch.std]}}
        self._spec(spec)

    def test_spec_layers(self):
        spec = {'0': {'weights': [torch.mean, torch.std],
                      'grad_out': [torch.mean]},
                '2': {'weights': [torch.mean],
                      'grad_out': [torch.mean, torch.std]}
                }
        self._spec(spec)

    def test_bylayer_plots(self):
        spec = {'0': {'weights': [torch.mean, torch.std], 'grad_out': [torch.mean]},
                '2': {'weights': [torch.mean], 'grad_out': [torch.mean, torch.std]}}
        plot_args, meter_args = get_preset('loss+MSE')
        self._spec_plus_add(spec, plot_args=plot_args, meter_args=meter_args, mode='by_layer')

    def test_custom_plots(self):
        spec = \
            {'plot1':
                 {'0': {'weights': [torch.std], 'grad_out': [torch.mean]},
                  '2': {'weights': [torch.mean], 'grad_out': [torch.mean]}},
             'plot2':
                 {'0': {'weights': [torch.std], 'grad_out': [torch.mean]},
                  '2': {'weights': [torch.mean], 'grad_out': [torch.mean, ]}}}

        plots =  {'plt1': {'type': 'line'}, 'plt2': {'type': 'line'}}
        meters= \
            [{'layer': '0', 'data': 'grad_out', 'func': torch.std, 'target': 'plt1'},
             {'layer': '0', 'data': 'grad_out', 'func': torch.mean, 'target': 'plt1'},
             {'layer': '2', 'data': 'grad_out', 'func': torch.mean, 'target': 'plt1'},
             {'layer': '2', 'data': 'grad_out', 'func': torch.std, 'target': 'plt1'} ]

        # plot_args, meter_args = get_preset('loss+MSE')
        self._spec_logger2(plot_args=plots, meter_args=meters)

    def test_custom_plots2(self):
        traceopts = dict(mode = 'markers', line=dict(dash = 'dot'))
        plots =  {'plt1': {'type': 'line', 'opts':
                             {
                              'traceopts':traceopts,
                              'markersize': 2
                             }},
                  'plt2': {'type': 'line'}}

        meters= \
            [{'layer': '0', 'data': 'grad_out', 'func': torch.std, 'target': 'plt1'},
             {'layer': '0', 'data': 'grad_out', 'func': torch.mean, 'target': 'plt1'},
             {'layer': '2', 'data': 'grad_out', 'func': torch.mean, 'target': 'plt1'},
             {'layer': '2', 'data': 'grad_out', 'func': torch.std, 'target': 'plt1'}
             # name-> 'layer_2'
             ]

        # plot_args, meter_args = get_preset('loss+MSE')
        self._spec_logger2(plot_args=plots, meter_args=meters)









