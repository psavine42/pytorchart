import pprint
from collections import defaultdict
from .preconfigured import FlexLogger
from .tooled import TooledModel, _default_metrics


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
    Usage:

    model = nn.Sequential(nn.Linear(20, 10), nn.Linear(10, 3))
    TM = ToolingLogger(model)

    """
    def __init__(self,
                 model,
                 plot_args=None,
                 meter_args=None,
                 metrics=_default_metrics,
                 mode='by_layer',
                 **kwargs):
        self._cfg = kwargs
        self._Logger = None
        self._TM = TooledModel(model, metrics=metrics)
        self._tm_to_logger = defaultdict()
        # self._ctr = 0
        if mode == 'by_layer':
            self.generete_plots_by_layer(self._TM.get_dict())
        elif mode == 'by_metric':
            self.generete_plots_by_metric(self._TM.get_dict())
        if plot_args is not None and meter_args is not None:
            self._Logger.update(plot_args, meter_args)

    def _name_layer_plot(self, layer_name):
        return 'layer_' + str(layer_name)

    def generete_plots_by_metric(self, data_dict):
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

    def generete_plots_by_layer(self, data_dict):
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

    def step(self, log=False, keys=None, reset=True):
        _step = self._Logger.step()
        update_dict = {}
        for meter_name in self._Logger.get_meter_names():

            name, mtype, metric = self._tm_to_logger.get(meter_name, ['', '', ''])
            value = self._TM.get_metrics(name, mtype, metric)

            if value and len(value) >= 1:
                update_dict[meter_name] = value[0]
        self._Logger.add(update_dict)
        if log is True:
            self._Logger.log(X=_step, keys=keys, reset=reset)

    def add(self):
        self.step()

    def log(self, **kwargs):
        self._Logger.log(**kwargs)

    def reset(self):
        self._Logger.reset()

    def clear(self):
        self._TM.clear()

    def table(self):
        self._TM.table()


class TooledModelLogger2(TooledModel, ):
    def __init__(self, *args, **kwargs):
        super(TooledModelLogger2, self).__init__(*args, **kwargs)
