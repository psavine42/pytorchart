import torch
import torch.nn as nn
import numpy as np
from logutils.Loggers.style_utils import _spec
import torch.nn.functional as F
from utils_plus import show
import pprint

# _base = {'layer': '2', 'data': 'grad_out', 'func': torch.mean, 'target': 'plt1',
#          'display': {'line': {'color': 'blue', 'dash': 'dash'}}}

_index = ['layer', 'name', 'data', 'func']


def get_in(o, kys, d=None):
    ob = o.copy()
    while ob and kys:
        k = kys.pop(0)
        ob = ob.get(k, None)
    if ob is None:
        return d
    return ob


def values(o, kys):
    return [o.get(k) for k in kys]


def identity(*x):
    return x


def summarize(model, fn=identity):
    res = []
    for key, module in model._modules.items():
        if type(module) in [
            nn.modules.container.Container,
            nn.modules.container.Sequential
        ]:
            res += summarize(module, fn=fn)
        else:
            summary = fn(key, module)
            if summary is not None:
                res.append(summary)
    return res


def flatten_indexed(dicts, indexes):
    """

    :param dicts:
    :param indexes:
    :return:

    Usage:
    indexes = ['name', 'word']
    mydict = {'bob': {'and': {'blah':1 }, 'was': {'cat':2 }}}

    flatten_indexed(mydict, indexes)
    >> [{'name':'bob', 'word':'and', 'blah':1 },
        {'name':'bob', 'word':'was', 'cat':2 }]
    """
    return


def name_type_summary(k, module):
    if isinstance(module, nn.Linear):
        return k
    return None


def rgb(*args):
    r, g, b = args[0], args[1], args[2]
    def clamp(x):
        return max(0, min(x, 255))
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))


#############################
def _grad_wrt_weight(module, grad_in):
    m = list(module._parameters['weight'].size())
    if type(module) in [nn.Linear]:
        m = m[::-1]
    if grad_in is not None:
        for g in grad_in:
            if g is not None and list(g.size()) == m:
                return g.data


def grad_mean_snr(module, grad_in, grad_out):
    m = module._parameters['weight'].data.max()
    g_wrt_w_data = _grad_wrt_weight(module, grad_in)
    n = g_wrt_w_data / m
    return n.mean()


def grad_std_snr(module, grad_in, grad_out):
    m = module._parameters['weight'].data.max()
    g_wrt_w_data = _grad_wrt_weight(module, grad_in)
    n = g_wrt_w_data / m
    return n.std()


def gen_grad_wrt_w(fn):
    def hook(module, grad_in, grad_out):
        return fn(_grad_wrt_weight(module, grad_in))
    return hook


def gen_module_wght(fn):
    def hook(module, grad_in, grad_out):
        return fn(module._parameters['weight'].data)
    return hook


fspecs = \
    {'snr':
         {'layer': lambda name, m: name if type(m) in [nn.Linear, nn.Conv2d] else None,
          'data': 'backward',
          'name': 'std_meter',
          'func': [grad_mean_snr, grad_std_snr],
          'same': {'layer': 'line.color',
                   'func': 'line.dash'}}}


def SNR(model, **kwargs):
    data = []
    spec = fspecs.get('snr')
    target_plt = kwargs.get('target', 'plot')
    funcs = spec['func']
    _sumary = summarize(model, fn=spec['layer'])
    colors = np.random.randint(0, 255, (len(_sumary), 3,))
    styles = _spec.get('line.dash')
    if kwargs.get('debug', None) is True:
        print(_sumary)

    for i_l, module_name in enumerate(_sumary):
        for i_f, fn in enumerate(funcs):
            res = {'layer': module_name,
                   'data': spec['data'],
                   'target': target_plt,
                   'func': fn,
                   'display': {'line': {'color': rgb(*colors[i_l]),
                                        'dash': styles[i_f]}}}
            data.append(res)
    return data




