import time, pickle
from inspect import signature
from torchnet import meter as METERS
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from collections import defaultdict
from .meter_doc import _meter_defs


_meters = list(_meter_defs.keys())
_plots = ['line', 'scatter']
_loggers = ['heatmap'
            # image
            ]
_modes = ['train', 'valid', 'test']


class FlexLogger(object):
    def __init__(self, plot_args, meter_args, **kwargs):
        """

        :param plot_args: dictionary of definitions for plotters
        :param meter_args: dictionary of definitions for meters
        :param meter_args: dictionary of definitions for meters
        :Attributes:
            _meters:
            _plots :
            _links : map of { meters: plots }

        Usages:
          ReportLogger(
            {'loss': {'type': 'line',  'opts': [] } }

            {'train_loss': {'type': 'averagevaluemeter',  'target': 'loss' },
             'test_loss':  {'type': 'averagevaluemeter',  'target': 'loss' },  })

        """
        # saving a copy of args for now
        self._args = {'meter': meter_args, 'plot': plot_args}
        self._env = kwargs.get('env', None)
        self._guid = kwargs.get('uid', str(time.time()).split('.')[0])
        self._port = kwargs.get('port', 8097)
        self._epoch = 0

        self._meters = {}
        self._plots = {}

        self._plot_to_meter = defaultdict(list)
        self._meter_to_plot = defaultdict(str)

        # initialization
        self.update_config(plot_args, meter_args)

    def update_config(self, plot_args, meter_args):
        self._init_links(plot_args, meter_args)
        self._init_plots(plot_args)
        self._init_meters(meter_args)

    def _init_links(self, plot_args, meter_args):
        for k, v in meter_args.items():
            target_plot = v.get('target', '')
            self._meter_to_plot[k] = target_plot
            self._plot_to_meter[target_plot].append(k)

    def _init_meters(self, meter_args):
        assert isinstance(meter_args, dict), 'meters not defined as map'
        for name, v in meter_args.items():
            assert isinstance(v, dict), 'meter {} is not map'.format(name)
            self._add_meter(name, v)

    def _init_plots(self, plot_args):
        assert isinstance(plot_args, dict), 'plots not defined as map'
        for name, v in plot_args.items():
            assert isinstance(v, dict), 'plot {} is not map'.format(name)
            self._add_plot(name, v)

    def _add_plot(self, name, v):
        plot_type = v.pop('type', None)
        if plot_type is None:
            return
        port = v.pop('port', self._port)
        env = v.pop('env', self._env)
        opts = v.get('opts', {})

        # set legend to be indexed by corresponding meters
        opts['legend'] = self._plot_to_meter.get(name)
        opts['title'] =  v.get('title', name).capitalize()

        # setup class
        Klass = VisdomPlotLogger if plot_type in _plots else VisdomLogger
        self._plots[name] = Klass(plot_type, port=port, opts=opts, env=env)

    def _add_meter(self, name, v):
        meter_type = v.get('type', None)
        # target_plot = v.get('target', None)
        opts = v.get('opts', None)

        Klass = METERS.__dict__.get(meter_type, None)
        if Klass is None:
            print('cannot initialize ', Klass)
            return
        n_args = len(signature(Klass.__init__).parameters)
        if n_args == 1:
            self._meters[name] = Klass()
        elif n_args > 1 and isinstance(opts, list):
            self._meters[name] = Klass(*opts)
        elif n_args > 1 and isinstance(opts, dict):
            self._meters[name] = Klass(**opts)
        else:
            self._meters[name] = Klass()

    def _prep_key_args(self, keys, items):
        if keys is None:
            keys = list(items.keys())
        elif isinstance(keys, str):
            keys = [keys]
        return keys

    def add(self, kwargs={}):
        """

        :param kwargs:
        :return:
        """
        for k, v in kwargs.items():
            if k not in self._meters:
                print('Meter not found ', k)
                continue
            if isinstance(v, float):
                self._meters.get(k).add(v)
            else:
                self._meters.get(k).add(*v)

    def log(self, X=None, keys=None, reset=True):
        """

        :param keys: X integer - X axis Value
        :param keys: list of names of plots
        :param reset: reset meters after plotting
        :return:
        """
        plot_keys = self._prep_key_args(keys, self._plots)
        X = self._epoch if X is  None else X
        for plot_ky in plot_keys:
            plot = self._plots.get(plot_ky, None)
            if plot is None:
                print('Key not found ', plot_ky)
                continue
            YS = []   # get the meters
            for meter_key in self._plot_to_meter.get(plot_ky, []):
                meter = self._meters.get(meter_key)
                val = meter.value()
                if isinstance(val, float):
                    YS.append(val)
                else:
                    YS.append(val[0])
                if reset is True:
                    meter.reset()
            if YS:
                XS = [X] * len(YS)
                plot.log(XS, YS)

    def save(self, file_path, plots=False):
        """
        saves this object, and the visdom state if plots is True
        todo implement lol
        :return:
        """
        # raise NotImplementedError('not yet implemented')
        pickle.dump(self, file_path)

    @staticmethod
    def load(file_path):
        obj = pickle.load(file_path)
        return obj

    def reset(self, keys=None):
        keys = self._prep_key_args(keys, self._meters)
        for k in keys:
            meter = self._meters.get(k, None)
            if meter is not None:
                meter.reset()

    def set_epoch(self, epoch=None):
        if epoch is None:
            self._epoch += 1
        else:
            self._epoch = epoch

    def remove_configs(self, keys):
        for k in keys:
            self._meters.pop(k, None)
            self._plots.pop(k, None)
            self._plot_to_meter.pop(k, None)
            self._meter_to_plot.pop(k, None)

    def get_plot_names(self):
        return list(self._plot_to_meter.keys())

    def get_plot_definitions(self):
        return self._plot_to_meter

    def get_meter_names(self):
        return list(self._meter_to_plot.keys())

    def get_meters_for_plot(self, plot_key):
        return self._plot_to_meter.get(plot_key, [])


class ModalLogger(FlexLogger):
    def __init__(self, *args, **kwargs):
        super(ModalLogger, self).__init__(*args, **kwargs)








