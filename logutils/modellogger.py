from collections import defaultdict
from .flexilogger import FlexLogger
from .tooledmodel import TooledModel


class TooledModelLogger(object):
    """

    Combines a flexlogger and a TooledModel.
    At each step, sends data from TooledModel to Logger.
        {'plots':
            {'layer_1': {'type': 'line'},
             'layer_2': {'type': 'line' }},
         'meters':
            {'layer_1_mean': {'type': 'AverageValueMeter', 'target': 'layer_1',
             'layer_1_std' : {'type': 'AverageValueMeter', 'target': 'layer_1',
             'layer_2_mean': {'type': 'AverageValueMeter', 'target': 'layer_2',
             'layer_2_std' : {'type': 'AverageValueMeter', 'target': 'layer_2'},
            }
         }

    Attribs:

    _tm_to_logger: index of TooledModel storages to paths in logger plot space.
        {'layer_0_grad_out_mean': ['0', 'grad_out', 'mean'],
         'layer_0_weights_std': ['0', 'weights', 'std'],
         'layer_2_weights_mean': ['2', 'weights', 'mean']})


    Usage:

    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
    TM = ToolingLogger(model)

    """
    def __init__(self,
                 model,
                 plot_args=None,
                 meter_args=None,
                 metrics=None,
                 spec=None,
                 mode='by_layer',
                 **kwargs):
        self._cfg = kwargs
        self._Logger = None
        self._TM = TooledModel(model, metrics=metrics, spec=spec)
        self._tm_to_logger = defaultdict()
        # self._ctr = 0
        if mode == 'by_layer':
            self._generete_plots_by_layer(self._TM.get_dict())
        elif mode == 'by_metric':
            self._generete_plots_by_metric(self._TM.get_dict())
        if plot_args is not None and meter_args is not None:
            self._Logger.update_config(plot_args, meter_args)

    def _name_layer_plot(self, layer_name):
        return 'layer_' + str(layer_name)

    def _generete_plots_by_metric(self, data_dict):
        """
        todo implement
        :param data_dict:
        :return:
        """
        raise NotImplementedError('not implemented')
        plots, meters = {}, {}
        for k, metrics_types in data_dict.items():
            layer_plot = self._name_layer_plot(k)
            plots[layer_plot] = {'type': 'line'}
            for metric_type, metrics in metrics_types.items():
                for metric in metrics.keys():

                    meter_name = metric_type + '_' + metric
                    meters[meter_name] = {'type': 'AverageValueMeter', 'target': layer_plot}
                    self._tm_to_logger[meter_name] = [k, metric_type, metric]
        self._Logger = FlexLogger(plots, meters)

    def _generete_plots_by_layer(self, data_dict):
        plots, meters = {}, {}
        for k, metrics_types in data_dict.items():
            layer_plot = self._name_layer_plot(k)
            plots[layer_plot] = {'type': 'line'}
            for metric_type, metrics in metrics_types.items():
                for metric in metrics.keys():
                    meter_name = layer_plot + '_' + metric_type + '_' + metric
                    meters[meter_name] = {'type': 'AverageValueMeter', 'target': layer_plot}
                    self._tm_to_logger[meter_name] = [k, metric_type, metric]
        self._Logger = FlexLogger(plots, meters)

    def generete_custom_plot(self, data_dict, plots, meters):
        """

        :param data_dict:
        :param plots:
        :param meters:
        :return:
        """
        self._Logger = FlexLogger(plots, meters)

    def step(self, X=None, log=False, keys=None, reset=True):
        """

        :param X:
        :param log:
        :param keys:
        :param reset:
        :return:
        """
        update_dict = {}
        for meter_name in self._Logger.get_meter_names():
            path = self._tm_to_logger.get(meter_name, None)
            if path is None:
                continue
            value = self._TM.get_metrics(*path)
            if value and len(value) >= 1:
                update_dict[meter_name] = value[0]

        self._Logger.add(update_dict)
        if log is True:
            self._Logger.log(X=X, keys=keys, reset=reset)

    # Passthrough APIs - should I just subclass?
    def add(self, update_dict):
        self._Logger.add(update_dict)

    def log(self, **kwargs):
        self._Logger.log(**kwargs)

    def reset(self):
        self._Logger.reset()

    def clear(self):
        self._TM.clear()

    def table(self):
        self._TM.table()

    def get_handles(self):
        return self._TM.get_handles()

    def get_dict(self):
        return self._TM.get_dict()


