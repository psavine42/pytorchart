import time, pickle
from inspect import signature
from torchnet import meter as METERS
from collections import defaultdict
from .meter_doc import _meter_defs
from .Loggers import TraceLogger


_meters = list(_meter_defs.keys())
_plots = ['line', 'scatter']
_loggers = ['heatmap'
            # image
            ]
_modes = ['train', 'valid', 'test']


class FlexLogger:
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
        self._env  = kwargs.get('env', None)
        self._guid = kwargs.get('uid', str(time.time()).split('.')[0])
        self._port = kwargs.get('port', 8097)
        self._step = kwargs.get('track_step', False)
        self._ctr  = kwargs.get('step', 0)
        self.debug = kwargs.get('debug', False)
        # hold meters and plots
        self._meters = defaultdict(dict)
        self._plots  = defaultdict(dict)

        # indexes
        self._plot_to_meter = defaultdict(list)
        self._meter_to_plot = defaultdict(str)

        # initialization
        self.update_config(plot_args, meter_args)

    def update_config(self, plot_args, meter_args):
        """
        Api for adding meters and plots
        :param plot_args:
        :param meter_args:
        :return:
        """
        self._init_links(meter_args)
        self._init_meters(meter_args)
        self._init_plots(plot_args)

    def _init_links(self, meter_args):
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
        for name, args in plot_args.items():
            assert isinstance(args, dict), 'plot {} is not map'.format(name)
            self._add_plot(name, args)

    def _add_plot(self, name, args):
        """
        create visdom Plot and add to dictionary
        :param name:
        :param args:
        :return:
        """
        plot_type = args.pop('type', None)
        if plot_type is None:
            print('invalid plot type')
            return
        port = args.pop('port', self._port)
        env  = args.pop('env', self._env)
        opts = args.get('opts', {})

        # set legend to be indexed by corresponding meters
        opts['legend'] = sorted(self._plot_to_meter.get(name, []))
        opts['title']  = args.get('title', name)

        traces = {}
        for mtr_name in opts['legend']:
            plot_opts = self._meters.get(mtr_name, {}).get('meta', {}).get('display', {})
            plot_opts['name'] = mtr_name
            traces[mtr_name] = plot_opts
        opts['data'] = traces
        self._plots[name]['obj'] = TraceLogger(
            title=opts['legend'], port=port, opts=opts, env=env)
        self._plots[name]['meta'] = opts # {**opts, **args}

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
            keys = sorted(list(items.keys()))
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
            if type(v) in [int, float]:
                self._meters.get(k).get('obj').add(v)
            else:
                self._meters.get(k).get('obj').add(*v)

    def log(self, X=None, keys=None, reset=False, step=False):
        """

        :param X: X integer - X axis Value
        :param keys: list of names of plots
        :param reset: reset meters after plotting
        :return:
        """
        plot_keys = self._prep_key_args(keys, self._plots)
        X = self._ctr if X is None or self._step is True else X
        for plot_ky in plot_keys:
            plot = self._plots.get(plot_ky, {}).get('obj', None)
            if plot is None:
                print('Key not found ', plot_ky)
                continue
            YS = []   # get the meters
            for meter_key in self._plot_to_meter.get(plot_ky, []):
                meter = self._meters.get(meter_key, {}).get('obj')
                val = meter.value()
                if isinstance(val, float):
                    YS.append(val)
                else:
                    YS.append(val[0])
                if reset is True:
                    meter.reset()
            YS = list(filter(lambda v: v == v, YS))
            if YS:
                XS = [X] * len(YS)
                plot.log(XS, YS)
        if step is True:
            self.step()

    def step(self, step=None):
        if step is None:
            self._ctr += 1
        else:
            self._ctr = step
        return self._ctr

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

    def show(self, meta=False):
        st = '\n'
        st += 'Step: {}\n'.format(self._ctr)
        st += 'Plots:\n'
        for plot, meters in self._plot_to_meter.items():
            st += ' ' * 2 + plot.capitalize()
            if meta is True:
                _meta = self._plots.get(plot, {}).get('meta', '')
                st += ' ' * 3 + 'meta: ' + str(_meta)
            st += '\n'
            for m in meters:
                mtr = self._meters.get(m, {}).get('obj')
                if mtr is None:
                    vl, cls = 0, 'na'
                else:
                    vl = mtr.value()
                    vl = vl if isinstance(vl, float) else vl[0]
                    cls = mtr.__class__.__name__.capitalize()
                st += ' ' * 6 + '{} - {} : {:.4f}\n'.format(m.capitalize().ljust(12, ' '), cls, vl)
                if meta is True:
                    _meta = self._meters.get(m, {}).get('meta', '')
                    st += ' ' * 9 + 'meta: ' + str(_meta) + '\n'
        return st

    def __call__(self, *args, **kwargs):
        self.add(kwargs)

    def __repr__(self):
        return self.show(meta=False)










