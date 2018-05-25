"""
Module for the FlexLogger Class
"""
import time, pickle, visdom
from inspect import signature
from torchnet import meter as METERS
from collections import defaultdict
from .meter_doc import meter_defs
from .Loggers import TraceLogger
import torch, pprint


_meters = list(meter_defs.keys())
_plots = ['line', 'scatter']
_loggers = ['heatmap'
            # image
            ]


class FlexLogger:
    """
    Base object for logging. It takes some specifications for meters
    and for plots, indexes them, and adds a hook to send data to plots on log.

    :param plot_args: dictionary of definitions for plotters or None.
    :param meter_args: dictionary of definitions for meters
    :param kwargs: additional keyword arguments


    Examples:

    .. code-block:: python

        meters = {
            'mymetric': {'type': 'AverageValueMeter', 'target': 'misc'}
            'test_loss': {'type': 'AverageValueMeter', 'target': 'loss'}
            'train_loss': {'type': 'AverageValueMeter', 'target': 'loss'} # target is the plot key
        }
        plots = {'loss': {'type': 'line'}, 'misc': {'type': 'line'}}

        TM = FlexLogger(plots, meters, env='bob') # initializes plots with meters to visdom env 'bob'

        # sample expirement step - for more see unittests
        TM(mymetric=69, test_loss=0.94)
        TM.step()

        TM.log(reset=True) # log and reset

    """
    def __init__(self, plot_args, meter_args, **kwargs):
        # saving a copy of args for now
        self._args = {'meter': meter_args, 'plot': plot_args}
        self._env  = kwargs.get('env', None)
        self._guid = kwargs.get('uid', str(time.time()).split('.')[0])
        self._port = kwargs.get('port', 8097)
        self._step = kwargs.get('track_step', False)
        self._ctr  = kwargs.get('step', 0)
        self.debug = kwargs.get('debug', False)
        self._viz = visdom.Visdom()
        # hold meters and plots
        self._meters = defaultdict(dict)
        self._plots  = defaultdict(dict)
        # indexes
        self._phase_to_meter = defaultdict(str)
        self._plot_to_meter = defaultdict(list)
        self._meter_to_plot = defaultdict(str)

        # initialization
        self.update_config(plot_args, meter_args)

    def update_config(self, plot_args, meter_args):
        """
        Api for adding meters and plots.

        :param plot_args:
        :param meter_args:
        :return:
        """
        self._init_links(meter_args)
        self._init_meters(meter_args)
        self._init_plots(plot_args)

    def _add_link(self, k, mtr):
        target_plot = mtr.get('target', '')
        self._meter_to_plot[k] = target_plot
        self._plot_to_meter[target_plot].append(k)

    def _init_links(self, meter_args):
        for k, v in meter_args.items():
            self._add_link(k, v)

    def _init_meters(self, meter_args):
        assert isinstance(meter_args, dict), 'meters not defined as map'
        for name, v in meter_args.items():
            assert isinstance(v, dict), 'meter {} is not map'.format(name)
            self._add_meter(name, v)

    def _init_plots(self, plot_args):
        assert isinstance(plot_args, dict), 'plots not defined as map'
        for name, args in plot_args.items():
            self._add_plot(name, args)

    def _add_plot(self, name, args):
        """
        create visdom Plot and add to dictionary

        :param name: (str) name of plot
        :param args: (dict) options for plot
        :return:
        """
        plot_type = args.pop('type', None)
        if plot_type is None:
            print('invalid plot type', plot_type)
            return
        port = args.pop('port', self._port)
        env  = args.pop('env', self._env)
        opts = args.get('opts', {})

        # set legend to be indexed by corresponding meters
        opts['legend'] = self._plot_to_meter.get(name, [])
        opts['title']  = name # args.get('title', name)

        traces = {}
        for mtr_name in opts['legend']:
            plot_opts = self._meters.get(mtr_name, {}).get('meta', {}).get('display', {})
            plot_opts['name'] = mtr_name
            traces[mtr_name] = plot_opts
        opts['data'] = traces
        self._plots[name]['meta'] = opts
        self._plots[name]['obj'] = TraceLogger(
            legend=opts['legend'], port=port, opts=opts, env=env, vis=self._viz)

    def _add_meter(self, name, args):
        meter_type = args.get('type', 'AverageValueMeter')
        opts = args.get('opts', None)
        Klass = METERS.__dict__.get(meter_type, None)
        if Klass is None:
            print('cannot initialize ', Klass)
            return
        n_args = len(signature(Klass.__init__).parameters)
        if n_args > 1 and isinstance(opts, list):
            self._meters[name]['obj'] = Klass(*opts)
        elif n_args > 1 and isinstance(opts, dict):
            self._meters[name]['obj'] = Klass(**opts)
        else:
            self._meters[name]['obj'] = Klass()
        self._meters[name]['meta'] = args

    def _prep_key_args(self, keys, items):
        if keys is None:
            keys = list(items.keys())
        elif isinstance(keys, str):
            keys = [keys]
        return keys

    def add(self, kwargs={}):
        """
        Add a dictionary of values to meters.

        :param kwargs:
        :return:
        """
        for k, v in kwargs.items():
            if k not in self._meters:
                print('Meter not found ', k)
                continue
            if type(v) in [int, float]:
                self._meters.get(k).get('obj').add(v)
            elif type(v) in [list, tuple]:
                self._meters.get(k).get('obj').add(*v)

    def log(self, keys=None, reset=False, step=False):
        """
        Retrieves current values of all meters, and plots at current timestep.
        log is used to keep familiarity with torchnet interface.

        if the reset keyword is set to True, calls self.reset()
        if the step keyword is set to True, calls self.step()

        :param keys: list of names of plots or None. If None, plots all keys.
        :param reset: reset meters after plotting
        :param step: increment counter after plotting
        :return: None

        :Example:

        """
        plot_keys = self._prep_key_args(keys, self._plots)
        X = self._ctr # if X is None or self._step is True else X
        for plot_ky in plot_keys:
            plot = self._plots.get(plot_ky, {}).get('obj', None)
            if plot is None:
                print('Key not found ', plot_ky)
                continue
            YS = []   # get the meters
            kys = self._plot_to_meter.get(plot_ky, [])
            for meter_key in kys:
                meter = self._meters.get(meter_key, {}).get('obj')
                val = meter.value()
                if type(val) in [int, float]:
                    YS.append(val)
                elif type(val) in [tuple, list]:
                    YS.append(val[0])
                else:
                    YS.append(None)
                if reset is True:
                    meter.reset(keys)
            if YS:
                XS = [X] * len(YS)
                plot.log(XS, YS)
        if step is True:
            self.step()

    def step(self, step=None):
        """
        Increments the internal counter, or sets to value of :step arg

        :param step: (int) if step is specified, sets the internal counter to that step
        :return: (int) updated step count
        """
        if step is None:
            self._ctr += 1
        else:
            assert isinstance(step, int), 'step must be integer, got {}'.format(type(step))
            self._ctr = step
        return self._ctr

    def save(self, file_path, plots=False):
        """
        saves this object, and the visdom state if plots is True
        todo implement lol
        :return: None
        """
        raise NotImplementedError('not yet implemented')
        pickle.dump(self, file_path)

    @staticmethod
    def load(file_path):
        obj = pickle.load(file_path)
        return obj

    def reset(self, keys=None):
        """
        Resets all of own meters. If keys kwd is specified,
        only resets the meters with those key names.

        :param keys: list[str] list of keys which will be reset.
        :return: None
        """
        keys = self._prep_key_args(keys, self._meters)
        for k in keys:
            meter = self._meters.get(k, {}).get('obj')
            if meter is not None:
                meter.reset()

    def value(self, keys=None):
        """


        :param keys:
        :return:
        """
        keys = self._prep_key_args(keys, self._meters)
        mp = []
        for k in keys:
            meter = self._meters.get(k, {}).get('obj')
            if meter is not None:
                val = meter.value()
                if type(val) in [int, float, torch.Tensor]:
                    mp.append(val)
                elif type(val) in [tuple, list]:
                    mp.append(val[0])
                else:
                    mp.append(None)
            else:
                print('missing meter with key ', k)
                mp.append(None)
        return mp

    def add_presets(self, *args):
        """

        :param args:
        :return:
        """
        from . import Config
        self.update_config(*Config.get_presets(*args))

    def add_metrics_for(self, *args, plot=None, phases=None):
        """
        Adds some metrics to hte meter, with an optional target plot.
        If the plot already exists, they will be added there.

        :param args: list of strings corresponding to metrics
        :param plot: string - name of plot
        :param phases: list of strings corresponding to phases
        :return:

        Example:

        .. code-block:: python

            # create new flexlogger
            TM = FlexLogger.from_presets('acc')

            # ... do stuff

            # latter on, need to add some stuff to a new plot
            TM.add_metrics_for('loss', 'norm_loss', 'hinge_loss', 'demorgan_loss', plot='loss', phases=['test])

        """
        assert plot is not None
        from .presets import Config
        self.update_config(*Config.gen_plot(*args, plot=plot, phases=phases))

    @classmethod
    def from_presets(cls, *args, phases=None):
        """
        Factory method to generate a logger from some preconfigured keys.
        see presets.preconfigured.Config for details

        :param args:
        :param phases:
        :return: instance of Flexilogger

        :Example:

        .. code-block:: python

            TM = FlexLogger.from_presets('acc', 'loss')

        """
        from .presets import Config
        return cls(*Config.get_presets(*args, phases=phases))

    @property
    def vis(self):
        """
        property: Retrieve the Visdom() object for fun or profit

        :return: Visdom() object
        """
        return self._viz

    def show(self, meta=False):
        """
        Implementation for __repr__ with additional functionality.
        __repr__ shows only meters, but show gives options to show metadata
        for charts and meters.

        :param meta: (boolean)
        :return: (string) detailed representation of self.

        Example:

        .. code-block:: python

            # create Logger
            Stat = FlexLogger('loss', 'acc')
            Stat.show()

        Output:

        .. code-block:: console

            Plots:
                loss
                    train_loss   - AverageValueMeter : nan
                    test_loss    - AverageValueMeter : nan
                acc
                    test_acc     - AverageValueMeter : nan
                    train_acc    - AverageValueMeter : nan
                Not plotted:

        """
        seen, st = set(), '\n'

        def _show_meter(st, m):
            mtr = self._meters.get(m, {}).get('obj')
            if mtr is None:
                vl, cls = 0, 'na'
            else:
                vl = mtr.value()
                vl = vl if isinstance(vl, float) else vl[0]
                cls = mtr.__class__.__name__
            st += ' ' * 6 + '{} - {} : {:.4f}\n'.format(m.ljust(12, ' '), cls, vl)
            if meta is True:
                _meta = self._meters.get(m, {}).get('meta', '')
                st += ' ' * 9 + 'meta: ' + str(_meta) + '\n'
            return st

        st += 'Step: {}\n'.format(self._ctr)
        st += 'Plots:\n'
        for plot, meters in self._plot_to_meter.items():
            st += ' ' * 2 + plot
            if meta is True:
                _meta = self._plots.get(plot, {}).get('meta', '')
                st += ' ' * 3 + 'meta: ' + str(_meta)
            st += '\n'
            for m in meters:
                seen.add(m)
                st = _show_meter(st, m)

        st += ' ' * 2 + ' Not plotted:'
        for m in self._meters.keys():
            if m not in seen:
                st = _show_meter(st, m)
        return st

    def __add__(self, other):
        return self

    def __len__(self):
        return len(self._meter_to_plot)

    def __call__(self, *args, **kwargs):
        self.add(kwargs)

    def __repr__(self):
        return self.show(meta=False)


    # def get_plot_names(self):
    #     """
    #
    #     :return:
    #     """
    #     return list(self._plot_to_meter.keys())
    #
    # def get_plot_definitions(self):
    #     """
    #
    #     :return:
    #     """
    #     return self._plot_to_meter
    #
    # def get_meter_names(self):
    #     """
    #
    #     :return:
    #     """
    #     return list(self._meter_to_plot.keys())
    #
    # def get_meters_for_plot(self, plot_key):
    #     """
    #
    #     :param plot_key:
    #     :return:
    #     """
    #     return self._plot_to_meter.get(plot_key, [])
    #
    # def remove_configs(self, *keys):
    #     """
    #
    #     :param keys:
    #     :return:
    #     """
    #     for k in keys:
    #         self._meters.pop(k, None)
    #         self._plots.pop(k, None)
    #         self._plot_to_meter.pop(k, None)
    #         self._meter_to_plot.pop(k, None)





