import torch
import pprint
from collections import OrderedDict, defaultdict


_HOOKS = {'forward': {'inputs', 'weights', 'outputs'},
          'backward': {'grad_in', 'grad_out'}}

_forward_hooks = _HOOKS['forward']
_backwrd_hooks = _HOOKS['backward']

_default_metrics = ['grad_in', 'grad_out', 'weights', 'inputs']
_default_hooks = [torch.mean, torch.std]  # , torch.min, torch.max

# _metrics_dict = {'*fc': {'grad_out': [torch.mean, torch.std]}}


class TooledModel(object):
    """

    TM = TooledModel(model, record=['grad_in', 'weights' )

    How to specify all the things
    [layer: backward: grads_in : [ functions ] ]
                      grads_out : [ functions }
            forward:  module : { functions }

    For modules creates a tree:
    {'module_name' : {'grad_in' : [] }


    """
    def __init__(self,
                 model=None,
                 metrics=_default_metrics,
                 funcs=_default_hooks,
                 layers=None,
                 spec=None):
        """

        :param opts: list of options
        """
        self._opts = None # [o for o in metrics if o in _default_metrics]
        self._data = None
        self._handles = []
        self.reset()
        if model is not None and spec is None:
            self._opts = [o for o in metrics if o in _default_metrics]
            self.register_model(model, layers=layers, funcs=funcs)
        elif model is not None and spec is not None:
            self.register_model(model, spec=layers, funcs=funcs)

    @property
    def register_forward(self):
        return len(_forward_hooks.intersection(self._opts)) > 0

    @property
    def register_backward(self):
        return len(_backwrd_hooks.intersection(self._opts)) > 0

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

    def _fn_to_str(self, fn):
        if isinstance(fn, str):
            return fn
        else:
            return fn.__name__

    def _fwd_weights_hook(self, layer_name, funcs):
        def hook(module, input, output):
            for fn in funcs:
                state_summary = [fn(i) for i in list(module.state_dict().values()) if i is not None]
                self._data[layer_name]['weights'][fn.__name__] = state_summary
        return hook

    def _fwd_output_hook(self, layer_name, funcs):
        def hook(module, input, output):
            for fn in funcs:
                self._data[layer_name]['inputs'][fn.__name__] = [fn(i.data) for i in input if i is not None]
        return hook

    def _bkwd_grad_in_hook(self, layer_name, funcs):
        def hook(module, grad_in, grad_out):
            for fn in funcs:
                self._data[layer_name]['grad_in'][fn.__name__] = [fn(i.data) for i in grad_in if i is not None]
        return hook

    def _bkwd_grad_out_hook(self, layer_name, funcs):
        def hook(module, grad_in, grad_out):
            for fn in funcs:
                self._data[layer_name]['grad_out'][fn.__name__] = [fn(i.data) for i in grad_out if i is not None]
        return hook

    def _forward_io_hook(self, layer_name, opts, funcs):
        """
         Functor for Module weights and IO hooks
        :param layer_name:
        :param funcs:
        :return:
        """
        def hook(module, input, output):
            for f in funcs:
                f_name = f.__name__
                if 'weights' in opts:
                    state_summary = [f(i) for i in list(module.state_dict().values()) if i is not None]
                    self._data[layer_name]['weights'][f_name] = state_summary
                if 'inputs' in opts:
                    self._data[layer_name]['inputs'][f_name] = [f(i.data) for i in input if i is not None]
        return hook

    def _backward_io_hook(self, layer_name, opts, funcs=_default_hooks):
        """ Functor for gradient hooks """
        def hook(module, grad_in, grad_out):
            for f in funcs:
                f_name = f.__name__
                if 'grad_out' in opts:
                    self._data[layer_name]['grad_out'][f_name] = \
                        [f(i.data) for i in grad_out if i is not None]
                if 'grad_in' in opts:
                    self._data[layer_name]['grad_in'][f_name] = \
                        [f(i.data) for i in grad_in if i is not None]
        return hook

    def _add_layer_index(self, layer_name, funcs, opts=None):
        if opts is None:
            opts = self._opts
        if layer_name not in self._data:
            self._data[layer_name] = defaultdict(dict)
        for opt in opts:
            for func in funcs:
                self._data[layer_name][opt][self._fn_to_str(func)] = []

    def _register_layer_fwd(self, module, layer_name, funcs, opts, reg=True):
        if any(opts) is True and any(funcs) is True and reg is True:
            hook = self._forward_io_hook(layer_name, opts, funcs)
            handle = module.register_forward_hook(hook)
            self._handles.append(handle)

    def _register_layer_bwd(self, module, layer_name, funcs, opts, reg=True):
        if any(opts) is True and any(funcs) is True and reg is True:
            hook = self._backward_io_hook(layer_name, funcs)
            handle = module.register_backward_hook(hook)
            self._handles.append(handle)

    # Public Api
    def register_model_dict(self, model, spec):
        """

        :param model:
        :param spec:
        :return:

        Usage:
            my_spec = { 0:{'grad_out': [torch.mean], 'weights': [torch.std]}}

            model =  nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
            TM = TooledModel(model, spec=my_spec)

        """
        for layer_name, module in list(model._modules.items()):
            if layer_name in spec:
                hooks = spec.get(layer_name, {})
                opts = list(hooks.keys())
                fwd_fns = hooks.get('inputs', []) + hooks.get('weights', [])
                bwd_fns = hooks.get('grad_in', []) + hooks.get('grad_out', [])

                fwd_triggers = set(opts).intersection(_forward_hooks)
                bwd_triggers = set(opts).intersection(_backwrd_hooks)

                self._add_layer_index(layer_name, fwd_fns, fwd_triggers)
                self._add_layer_index(layer_name, bwd_fns, bwd_triggers)

                self._register_layer_fwd(module, layer_name, fwd_triggers, fwd_fns)
                self._register_layer_bwd(module, layer_name, bwd_triggers, bwd_fns)
                if len(module._modules) > 0:
                    self.register_model_dict(module, spec)

    def register_model(self, model, layers=None, funcs=None, spec=None):
        """

        :param model:
        :param layers:
        :param funcs:
        :return:


            my_spec = { 0:{'grad_out': [torch.mean], 'weights': [torch.std]}}

            model =  nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
            TM = TooledModel()
            TM.register_model(model, layers=None, funcs=[torch.mean])

        """
        if spec is not None:
            assert isinstance(spec, dict), 'spec is not dictionary'
            self.register_model_dict(model, spec)
            return
        for name, module in list(model._modules.items()):
            if layers is None or name in layers:
                self._add_layer_index(name, funcs, self._opts)
                self._register_layer_fwd(module, name, funcs, self._opts, self.register_forward)
                self._register_layer_bwd(module, name, funcs, self._opts, self.register_backward)
                if len(module._modules) > 0:
                    self.register_model(module, layers=layers, funcs=funcs)

    def remove_hooks(self):
        """

        :return:
        """
        for handle in self._handles:
            handle.remove()

    def get_dict(self):
        return self._data

    def clear(self):
        self._data = defaultdict(dict)
        self._handles = []
        self.remove_hooks()

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


    # def _register_layer(self, module, layer_name, fw_funcs, bw_funcs, fwd=True, bwd=True):
    #     if fwd is True:
    #         hook = self._forward_io_hook(layer_name, fw_funcs)
    #         handle = module.register_forward_hook(hook)
    #         self._handles.append(handle)
    #     if bwd is True:
    #         hook = self._backward_io_hook(layer_name, bw_funcs)
    #         handle = module.register_backward_hook(hook)
    #         self._handles.append(handle)