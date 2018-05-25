import torch
from . import FlexLogger
from torch.autograd import Variable


_HOOKS = {'forward':  ['weights', 'inputs',  'outputs'],
          'backward': ['na',      'grad_in', 'grad_out']}
_HOOKS2 = {'forward':  {'weights', 'inputs',  'outputs'},
           'backward': {'weights': [],
                        'grad_in': ['weights', 'bias', 'input'],
                        'grad_out': None}}

_forward_hooks = set(_HOOKS['forward'])
_backwrd_hooks = set(_HOOKS['backward'])
_full_fns = sorted(list(_HOOKS.keys()))
_datasources = list(_HOOKS.keys()) + list(_forward_hooks) + list(_backwrd_hooks)

_default_metrics = ['grad_in', 'grad_out', 'weights', 'inputs']
_default_hooks = [torch.mean, torch.std]


class FlexTooledModel(FlexLogger):
    """
    Logging a nn.Module values to a Flexlogger by adding hooks.

    nn.Modules natively give a forward and backward hook, which
    have a function signature of

    forward_hook(module, input, output) -> None

    backward_hook(module, grad_in, grad_out) -> grad

    This module will
      - take a set of specifications for such functions,
      - create hooks for them to nn.Module
      - connect the hooks to meters, and updated without the FlexLogger(data=xx) calls.
      - meters will per normal when using the FlexLogger.log()

    Example:



    """
    def __init__(self, plot_args, metrics, model=None, **kwargs):
        """

        :param kwargs: list of options
        """
        super(FlexTooledModel, self).__init__(plot_args, metrics, **kwargs)
        self._handles = []
        if model is not None:
            self.register_model(model)

    #######################

    def _init_links(self, meters):
        if isinstance(meters, dict):
            super(FlexTooledModel, self)._init_links(meters)
            return
        [self._add_link(self._layer_spec_to_name(m), m) for m in meters]

    def _init_meters(self, meters):
        if isinstance(meters, dict):
            super(FlexTooledModel, self)._init_meters(meters)
            return
        [self._add_meter(self._layer_spec_to_name(m), m) for m in meters]

    #######################
    def _get_datasource_index(self, datasource):
        """

        """
        if datasource in _full_fns:
            return datasource, -1
        elif datasource in _HOOKS['forward']:
            return 'forward', _HOOKS['forward'].index(datasource)
        elif datasource in _HOOKS['backward']:
            return 'backward', _HOOKS['backward'].index(datasource)
        else:
            print('invalid datasource ' + datasource)
            return None, None

    def _gen_module_hook(self, module, mtr_name, fn, datasource):
        def unwrap(x):
            if isinstance(x, Variable):
                x = x.data
            if torch.is_tensor(x) and x.numel() == 1:
                return x[0]
            return x

        direction, idx = self._get_datasource_index(datasource)

        def hook_part(*args):
            arg = args[idx]
            if isinstance(arg, tuple):
                self._meters[mtr_name]['obj'].add(fn(arg[0].data))
            elif isinstance(arg, Variable):
                self._meters[mtr_name]['obj'].add(fn(arg.data))
            return None

        def hook_full(*args):
            d = unwrap(fn(*args))
            self._meters[mtr_name]['obj'].add(d)
            return None

        def _wrt(module, ix, kys=['weights']):
            n_grads = ['weights', 'outputs']
            if module.bias is True:
                n_grads.insert(1, 'bias')

            idxs = []
            for k in kys:
                if k in n_grads:
                    idxs.append(n_grads.index(k))

            def _hook_full(*args):
                _args = [args[ix][i] for i in idxs]
                d = unwrap(fn(_args))
                self._meters[mtr_name]['obj'].add(d)
                return None
            return _hook_full

        if idx is None:
            return
        hook = hook_full if idx == -1 else hook_part

        if direction == 'forward':
            handle = module.register_forward_hook(hook)
        else:
            handle = module.register_backward_hook(hook)
        self._handles.append(handle)

    def _layer_spec_to_name(self, spec):
        name = spec.get('name', None)
        if not name and 'layer' in spec and 'func' in spec:
            lyer = spec.get('layer', None)
            data = spec.get('data', None)
            func = spec.get('func', None).__name__
            return '{}_{}_{}'.format(lyer, data, func)
        return name

    @classmethod
    def generate_model_dict(cls, model, meter_args, **kwargs):
        """
        turn a bunch of dictionaries
        """
        dicts = []
        for layer_name, module in list(model._modules.items()):
            for spec in meter_args:
                lkey = spec.get('layer', None)
                if callable(lkey):
                    use_layer = lkey(layer_name, module)
                else:
                    use_layer = lkey in layer_name
                data = spec.get('data', None)
                func = spec.get('func', None)
                if use_layer is False or func is None or data is None:
                    continue

            if len(module._modules) > 0:
                cls.generate_model_dict(module, meter_args, **kwargs)
        return dicts

    def register_model(self, model, step_on_first=True):
        """
        Registers a model to its hook functions. The hooks specs must
        already be registered in self

        :param model: nn.Module -
        :return:

        Usage:
            my_spec = { 0:{'grad_out': [torch.mean], 'weights': [torch.std]}}

            model =  nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
            TM = FlexTooledModel(model, spec=my_spec)

        """
        for layer_name, module in list(model._modules.items()):
            for mtr_name, spec in self._meters.items():
                spec_ = spec.get('meta', {})
                lkey = spec_.get('layer', None)
                if lkey is None or lkey != layer_name:
                    continue
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
        Dereferences all hooks from pytorch nn.Module

        :return: None
        """
        for handle in self._handles:
            handle.remove()

    def clear(self):
        """

        :return:
        """
        self.remove_hooks()
        self._handles = []

    def get_handles(self):
        """

        """
        return self._handles

