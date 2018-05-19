import torch
import torch.nn as nn
import unittest, pprint
from logutils import TooledModel,  TooledModelLogger, FlexTooledModel
from logutils.presets.preconfigured import get_preset
from torch.autograd import Variable
import torch.nn.functional as F
from logutils import functional as Fn
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(20, 10, bias=False)
        self.fc2 = nn.Linear(10, 3, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def gen_test_inputs():
    x = Variable(torch.rand(1, 20))
    targets = Variable(torch.rand(3, 1))
    return x, targets


def gen_net_inputs():
    x = Variable(torch.rand(1, 1, 28, 28).float())
    targets = Variable(torch.rand(1, 10))
    return x, targets


class TestFuncs(unittest.TestCase):
    def test_summary(self):
        def sumary_fn(k, module):
            return {k: type(module).__name__}
        net = Net()

        sum = Fn.summarize(net, sumary_fn)

        assert len(sum) == 5
        meter_defs = Fn.tishby(net)
        pprint.pprint(meter_defs)


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

    def _spec_logger2(self, meter_args=None, plot_args=None, model=Net1(), g=gen_test_inputs):
        optim = torch.optim.Adam(model.parameters())
        TM = FlexTooledModel(plot_args, meter_args, model, track_step=False)

        for i in range(4):
            optim.zero_grad()
            x, t = g()
            o = model(x)
            loss = F.mse_loss(o, t)

            TM.add()

            loss.backward()
            optim.step()
            #
            TM.log(reset=True, step=True)

    def exersize_TM(self, model, TM, g=gen_test_inputs, fn=None):
        optim = torch.optim.Adam(model.parameters())
        for i in range(10):
            optim.zero_grad()
            x, t = g()
            o = model(x)
            loss = F.mse_loss(o, t)
            if fn:
                fn(x, t, o, loss)

            loss.backward()
            optim.step()
            TM.log(reset=True, step=True)


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

    def test_tishby(self):
        plots = {'plt1': {'type': 'line'}}
        model = Net1()
        meters = Fn.SNR(model, target='plt1')
        self._spec_logger2(plot_args=plots, meter_args=meters, model=model)

    def test_multi_type(self):
        plots = {'plt1':
                    {'type': 'line',
                     'opts':
                        {'layout': {'ytype': 'log'}}}}
        model = Net()
        meters = Fn.SNR(model, target='plt1')
        TM = FlexTooledModel(plots, meters, model, track_step=False)

        plot, meter = get_preset('loss+MSE')
        TM.update_config(plot, meter)
        print(str(TM))

        def adfunc(x, t, out, loss):
            # print(loss.data[0])
            TM(train_loss=loss.data[0], train_mse=[t.data, out.data])

        self.exersize_TM(model, TM, g=gen_net_inputs, fn=adfunc)

    def test_conv(self):
        plots = {'plt1':
                    {'type': 'line',
                     'opts':
                        {'layout': {'ytype': 'log'}}}}
        model = Net()
        meters = Fn.SNR(model, target='plt1')
        TM = FlexTooledModel(plots, meters, model, track_step=False)

        plot, meter = get_preset('loss+MSE')
        TM.update_config(plot, meter)
        print(str(TM))

        def adfunc(x, t, out, loss):
            # print(loss.data[0])
            TM(train_loss=loss.data[0], train_mse=[t.data, out.data])

        self.exersize_TM(model, TM, g=gen_net_inputs, fn=adfunc)





