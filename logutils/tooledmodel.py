import torch
import pprint
from collections import defaultdict
from .modellogger import FlexLogger
from torch.autograd import Variable


_HOOKS = {'forward':  ['weights', 'inputs',  'outputs'],
          'backward': ['na',      'grad_in', 'grad_out']}
_forward_hooks = set(_HOOKS['forward'])
_backwrd_hooks = set(_HOOKS['backward'])
_full_fns = sorted(list(_HOOKS.keys()))
_datasources = list(_HOOKS.keys()) + list(_forward_hooks) + list(_backwrd_hooks)

_default_metrics = ['grad_in', 'grad_out', 'weights', 'inputs']
_default_hooks = [torch.mean, torch.std]


class FlexTooledModel(FlexLogger):
    """



    """
    def __init__(self, plot_args, metrics, model=None, **kwargs):
        """

        :param kwargs: list of options
        """
        super(FlexTooledModel, self).__init__(plot_args, metrics, **kwargs)
        self._handles = []
        if model is not None:
            self.register_model(model)

    def _init_links(self, meter_args):
        if isinstance(meter_args, dict):
            super(self)._init_links(meter_args)
            return
        for mtr in meter_args:
            target_plot = mtr.get('target', '')
            k = self._layer_spec_to_name(mtr)
            self._meter_to_plot[k] = target_plot
            self._plot_to_meter[target_plot].append(k)

    def _init_meters(self,  meter_args):
        if isinstance(meter_args, dict):
            print('NOT YET IMPLEMENTED')
            return # name=
        for spec in meter_args:
            assert isinstance(spec, dict), 'meter {} is not map'.format('')
            name = self._layer_spec_to_name(spec)
            self._add_meter(name, spec)

    def _get_datasource_index(self, datasource):
        if datasource in _full_fns:
            return _full_fns.index(datasource)
        elif datasource in _HOOKS['forward']:
            return 'forward', _HOOKS['forward'].index(datasource)
        elif datasource in _HOOKS['backward']:
            return 'backward', _HOOKS['backward'].index(datasource)
        else:
            print('invalid datasource ' + datasource)
            return None, None

    def _gen_module_hook(self, module, mtr_name, fn, datasource):
        def hook_func(*args):
            arg = args[idx]
            if isinstance(arg, tuple):
                self._meters[mtr_name]['obj'].add(fn(arg[0].data))
            elif isinstance(arg, Variable):
                self._meters[mtr_name]['obj'].add(fn(arg.data))

        direction, idx = self._get_datasource_index(datasource)
        if idx is None:
            return
        if direction == 'forward':
            handle = module.register_forward_hook(hook_func)
        else:
            handle = module.register_backward_hook(hook_func)
        self._handles.append(handle)

    def _layer_spec_to_name(self, spec):
        name = spec.get('name', None)
        if not name and 'layer' in spec and 'func' in spec:
            lyer = spec.get('layer', None)
            data = spec.get('data', None)
            func = spec.get('func', None).__name__
            return '{}_{}_{}'.format(lyer, data, func)
        return name

    # Public Api
    # todo - add register first option for stepping on first input
    def register_model(self, model, step_on_first=True):
        """

        :param model:
        :return:

        Usage:
            my_spec = { 0:{'grad_out': [torch.mean], 'weights': [torch.std]}}

            model =  nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
            TM = TooledModel(model, spec=my_spec)

        """
        for layer_name, module in list(model._modules.items()):
            for mtr_name, spec in self._meters.items():
                spec_ = spec.get('meta', {})
                lkey = spec_.get('layer', None)
                if lkey in layer_name:
                    data = spec_.get('data', None)
                    func = spec_.get('func', None)
                    if func is None or data is None:
                        print('Possible missing Definition ')
                        continue
                    self._gen_module_hook(module, mtr_name, func, data)
            if len(module._modules) > 0:
                self.register_model(module)

    def remove_hooks(self):
        """

        :return:
        """
        for handle in self._handles:
            handle.remove()

    def clear(self):
        self.remove_hooks()
        self._handles = []


    def get_handles(self):
        return self._handles



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
        self._opts = None
        self._data = None
        self._handles = []
        self.clear()
        if model is not None and spec is None:
            self._opts = [o for o in metrics if o in _default_metrics]
            self.register_model(model, layers=layers, funcs=funcs)
        elif model is not None and spec is not None:
            self.register_model_dict(model, spec=spec)

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

    def get_handles(self):
        return self._handles

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

    # all hooking functions
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
        # p#rint(opts)
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

    def _register_layer_fwd(self, module, layer_name, funcs, triggers, reg=True):
        if any(triggers) is True and any(funcs) is True and reg is True:
            hook = self._forward_io_hook(layer_name, triggers, funcs)
            handle = module.register_forward_hook(hook)
            self._handles.append(handle)

    def _register_layer_bwd(self, module, layer_name, funcs, triggers, reg=True):
        if any(triggers) is True and any(funcs) is True and reg is True:
            hook = self._backward_io_hook(layer_name, triggers, funcs)
            handle = module.register_backward_hook(hook)
            self._handles.append(handle)

    def _match_layer(self, layer, spec_kys):
        for k in spec_kys.keys():
            if k in layer:
                return spec_kys.get(k)
        return None

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
        _opts = set()
        for layer_name, module in list(model._modules.items()):
            hooks = self._match_layer(layer_name, spec)
            if hooks is not None:
                opts = list(hooks.keys())
                _opts.update(opts)
                fwd_fns = hooks.get('inputs', []) + hooks.get('weights', [])
                bwd_fns = hooks.get('grad_in', []) + hooks.get('grad_out', [])

                fwd_triggers = set(opts).intersection(_forward_hooks)
                bwd_triggers = set(opts).intersection(_backwrd_hooks)

                self._add_layer_index(layer_name, fwd_fns, fwd_triggers)
                self._add_layer_index(layer_name, bwd_fns, bwd_triggers)

                self._register_layer_fwd(module, layer_name, fwd_fns, fwd_triggers)
                self._register_layer_bwd(module, layer_name, bwd_fns, bwd_triggers)
                if len(module._modules) > 0:
                    self.register_model_dict(module, spec)
        self._opts = list(_opts)

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
                self._register_layer_fwd(module, name, funcs, self._opts, True)
                self._register_layer_bwd(module, name, funcs, self._opts, True)
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

