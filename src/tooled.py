import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
from inspect import signature
from .preconfigured import get_preset_logger

_modes = ['grad_in', 'grad_out', 'weights', 'inputs']


class TooledModel():
    """

    TM = TooledModel(model)

    For modules creates a tree:
    {'module_name' : {'grad_in',


    """
    def __init__(self, model=None, opts=_modes, keys=None):
        """

        :param opts: list of options
        """
        self._opts = [o for o in opts if o in _modes]
        self._grads_out = None
        self._grads_in = None
        self._handles = None
        self._io = None
        self.reset()
        if model is not None:
            self.register_model(model, keys=keys)

    def reset(self):
        self._grads_in = {}
        self._grads_out = {}
        self._io = {}
        self._handles = {}

    def grad_io_hook(self, name):
        """ Functor for gradient hooks """
        def hook(module, grad_input, grad_output):
            self._grads_in[name] = [i.data for i in grad_input if i is not None]
            self._grads_out[name] = [i.data for i in grad_output if i is not None]
        return hook

    def forward_io_hook(self, name):
        def hook(module, input, output):
            self._io[name] = [i.data for i in input if i is not None]
        return hook

    def register_model(self, model, keys=None):
        for name, module in list(model._modules.items()):
            if keys is None or name in keys:
                hook = self.grad_io_hook('g_io_' + name)
                handle = module.register_backward_hook(hook)

    def table(self):
        mxkey = max(map(len, self._grads_out.keys())) + 2
        for k, v in self._grads_out.items():
            print('{} {:.4f} {:.4f}'.format(k.ljust(mxkey, ' '), v[0].mean(), v[0].std()))

    def get_info(self):
        for k, v in self._grads_out.items():
             v[0].mean(), v[0].std()


class ToolingLogger():
    """

    Combines a flexlogger and a TooledModel.
    At each step, sends data from TooledModel to Logger.

    Usage:

    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
    TM = ToolingLogger(model)

    """
    def __init__(self, model, opts=_modes, **kwargs):
        self._cfg = kwargs.get('preset', 'grads')
        self._Logger = get_preset_logger(self._cfg)
        self._TM = TooledModel(model, opts=opts)

    def step(self):
        pass
