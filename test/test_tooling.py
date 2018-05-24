import torch
from torch.autograd import Variable
import unittest, pprint
from pytorchart import FlexTooledModel
from pytorchart.presets.preconfigured import get_preset, get_presets, Config
from pytorchart import functional as Fn

import random
from example.testmodels import *


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


    def test_model_log(self):
        model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(),  nn.Linear(10, 3))
        optim = torch.optim.Adam(model.parameters())
        TM = FlexTooledModel(model, metrics=['grad_out', 'weights'], funcs=[torch.mean])

        assert len(TM.get_handles()) == 6
        # assert TM._TM._opts == ['grad_out', 'weights']
        # # loop
        # for i in range(4):
        #     optim.zero_grad()
        #     x = Variable(torch.rand(2, 20))
        #     targets = Variable(torch.rand(2, 3))
        #
        #     o = model(x)
        #     loss = F.mse_loss(o, targets)
        #     TM.step(log=True)
        #     loss.backward()
        #     optim.step()
        # TM.table()

    def exersize_TM(self, model, TM, g=gen_test_inputs, fn=None, fnt=None):
        optim = torch.optim.Adam(model.parameters())
        for i in range(21):
            optim.zero_grad()
            x, t = g()
            o = model(x)
            loss = F.mse_loss(o, t)

            loss.backward()
            optim.step()
            if fn:
                # float('NaN')
                ls = float('NaN') if random.random() > 0.5 else loss.data[0]
                # ls = loss.data[0]
                # print(ls)
                fn(x, t, o, ls)   # train add

            # simulate testing
            if i % 2 == 0 and fnt:
                x, t = g()
                o = model(x)
                loss = F.mse_loss(o, t)
                fnt(x, t, o, loss.data[0])      # test add
                TM.log(reset=True, step=False)   # test add

            TM.step()       # step

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
        # self._spec_logger2(plot_args=plots, meter_args=meters)

    def test_tishby(self):
        plots = {'plt1': {'type': 'line'}}
        model = Net1()
        meters = Fn.SNR(model, target='plt1')
        self._spec_logger2(plot_args=plots, meter_args=meters, model=model)

    def test_multi_type(self):
        plots = {'plt1': {'type': 'line',
                          'opts': {'layout': {'ytype': 'log'}}}}
        model = Net()
        meters = Fn.SNR(model, target='plt1')
        TM = FlexTooledModel(plots, meters, model, track_step=False)

        plot, meter = get_presets('loss', 'mse')
        TM.update_config(plot, meter)
        print(str(TM))

        def adf(x, t, out, loss):
            TM(train_loss=loss, train_mse=[t.data, out.data])

        def testf(x, t, out, loss):
            TM(test_loss=loss, test_mse=[t.data, out.data])
        self.exersize_TM(model, TM, g=gen_net_inputs, fn=adf, fnt=testf)

    def test_conv(self):
        model = Net()
        layer_fn = lambda name, m: name if 'conv' in name else None
        meters, plots = Fn.generate_layers(model, fn=layer_fn,  targets=['grad_norms', 'snr', ])
        TM = FlexTooledModel(plots, meters, model, track_step=False)

        TM.update_config(*get_presets('mse', 'loss'))
        # TM.add_metrics_for('loss', 'loss2', plot='loss')
        # print(str(TM))

        def tf(x, t, out, loss): TM(train_loss=loss, train_mse=[t.data, out.data])
        def vf(x, t, out, loss): TM(test_loss=loss, test_mse=[t.data, out.data])
        self.exersize_TM(model, TM, g=gen_net_inputs, fn=tf, fnt=vf)

    def test_nolog(self):
        model = Net()
        meters, plots = Fn.generate_layers(model, targets=['grad_norms'])
        TM = FlexTooledModel(plots, meters, model, track_step=False)
        meters = get_meters('loss', 'mse')
        TM.update_config(None, meters)





