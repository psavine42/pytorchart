import torch
import torch.nn.functional as F
import pprint
from collections import OrderedDict, defaultdict
from .preconfigured import get_preset_logger, FlexLogger

_default_metrics = ['grad_in', 'grad_out', 'weights', 'inputs']
_default_hooks = [torch.mean, torch.std] # , torch.min, torch.max


class TooledModel():
    """

    TM = TooledModel(model, record=['grad_in', 'weights' )

    How to specify all the things
    [layer: backward: grads_in : [ functions ] ]
                      grads_out : [ functions }
            forward:  module : { functions }

    For modules creates a tree:
    {'module_name' : {'grad_in',


    """
    def __init__(self,
                 model=None,
                 metrics=_default_metrics,
                 layers=None,
                 funcs=_default_hooks):
        """

        :param opts: list of options
        """
        self._opts = [o for o in metrics if o in _default_metrics]
        self._funcs = funcs
        self._data = None
        self._handles = []
        self.reset()
        if model is not None:
            self.register_model(model, layers=layers, funcs=funcs)

    @property
    def register_forward(self):
        return len({'inputs', 'weights', 'outputs'}.intersection(self._opts)) > 0

    @property
    def register_backward(self):
        return len({'grad_in', 'grad_out'}.intersection(self._opts)) > 0

    @property
    def layers_with_metrics(self):
        return list(self._data.keys())

    @property
    def all_hooks(self):
        return self._opts

    @property
    def all_functions(self):
        func_dic = set()
        for layer, data in self._data.items():
            for metric_type in self._opts:
                metrics = data.get(metric_type, {})
                func_dic.add(list(metrics.keys()))
        return func_dic

    def reset(self):
        self._data = defaultdict(dict)
        self._forward_handles = {}
        self._backward_handles = {}

    def forward_io_hook(self, layer_name, funcs):
        """
         Functor for Module weights and IO hooks
        :param layer_name:
        :param funcs:
        :return:
        """
        def hook(module, input, output):
            for f in funcs:
                f_name = self._fn_to_str(f)
                if 'weights' in self._opts:
                    state_summary = [f(i) for i in list(module.state_dict().values()) if i is not None]
                    self._data[layer_name]['weights'][f_name] = state_summary
                if 'inputs' in self._opts:
                    self._data[layer_name]['inputs'][f_name] = [f(i.data) for i in input if i is not None]
        return hook

    def backward_io_hook(self, layer_name, funcs=_default_hooks):
        """ Functor for gradient hooks """
        def hook(module, grad_in, grad_out):
            for f in funcs:
                f_name = f.__name__
                if 'grad_out' in self._opts:
                    self._data[layer_name]['grad_out'][f_name] = \
                        [f(i.data) for i in grad_out if i is not None]
                if 'grad_in' in self._opts:
                    self._data[layer_name]['grad_in'][f_name] = \
                        [f(i.data) for i in grad_in if i is not None]
        return hook

    def add_layer(self, layer_name, funcs):
        self._data[layer_name] = defaultdict(dict)
        for opt in self._opts:
            for func in funcs:
                self._data[layer_name][opt][self._fn_to_str(func)] = []

    def register_model(self, model, layers=None, funcs=None):
        """

        :param model:
        :param layers:
        :param funcs:
        :return:
        """
        if funcs is None:
            funcs = self._funcs
        for name, module in list(model._modules.items()):
            if layers is None or name in layers:
                self.add_layer(name, funcs)

                if self.register_forward is True:
                    hook = self.forward_io_hook(name, funcs)
                    handle = module.register_forward_hook(hook)
                    self._handles.append(handle)

                if self.register_backward is True:
                    hook = self.backward_io_hook(name, funcs)
                    handle = module.register_backward_hook(hook)
                    self._handles.append(handle)

                if len(module._modules) > 0:
                    self.register_model(module, layers=layers, funcs=funcs)

    def remove_hooks(self):
        for handle in self._handles:
            handle.remove()

    def get_dict(self):
        return self._data

    def _fn_to_str(self, fn):
        if isinstance(fn, str):
            return fn
        else:
            return fn.__name__

    def get_metrics(self, layer, mtype, metric_name=None):
        metrics = self._data.get(layer, {}).get(mtype, {})
        if metric_name is None:
            return metrics
        else:
            metric_n = self._fn_to_str(metric_name)
            return metrics.get(metric_n, None)

    def table(self):
        num_val = 9
        mxkey   = max(num_val, max(map(len, list(self._data.keys()))) + 2)
        groups  = {x: 0  for x in self._opts}
        headers = {x: set() for x in self._opts}

        for layer, data in self._data.items():
            for metric_type in self._opts:
                metrics = self.get_metrics(layer, metric_type) # [layer, type, {}]
                num_items = len(metrics) * num_val
                headers[metric_type].update(set(metrics.keys()))
                groups[metric_type] = num_items if num_items > groups[metric_type] else groups[metric_type]

        header1 = ' | ' + ''.ljust(mxkey, ' ')
        header2 = ' | ' + 'Layers'.ljust(mxkey)
        header3 = ' +-' + ''.ljust(mxkey, '-')
        for k in sorted(list(groups.keys())):
            header1 += '|' + k.capitalize().center(groups.get(k), ' ')
            header2 += '|'
            header3 += '+'.ljust(groups.get(k)+1, '-')
            for name in sorted(list(headers.get(k))):
                header2 += name.center(num_val, ' ')

        header1 += '|'
        header2 += '|'
        header3 += '+'
        res = '\n' + header3 + '\n' + header1 + '\n' + header2 + '\n' + header3 + '\n'
        for layer, data in self._data.items():
            st = ' | ' + layer.ljust(mxkey, ' ')
            for metric_type in sorted(self._opts):
                st += '|'
                metric_dict = self.get_metrics(layer, metric_type)
                if not metric_dict:
                    st += ' ' * num_val
                else:
                    for metric, value in metric_dict.items():
                        if value and len(value) >= 1:
                            st += '{:.4f}'.format(value[0]).center(num_val, ' ')
                        else:
                            st += ''.center(num_val, ' ')

            res += st + '|\n'
        res += header3
        print(res)

    def __repr__(self):
        return str(self._data)

    def get_info(self):
        #for summary in self.summaries:
        pass


