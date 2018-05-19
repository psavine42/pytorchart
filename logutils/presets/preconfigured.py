from logutils.flexilogger import FlexLogger, _meters, _meter_defs
import numpy as np

_alt_defs = \
    {'simple':
        {'plots':
            {'loss':
                {'type': 'line',
                 'opts': {},
                 'meters':
                    {'train_loss': {'type': 'AverageValueMeter'},
                     'test_loss': {'type': 'AverageValueMeter'}}}},

         },
     'loss+MSE':
        {'plots':
            {'loss': {'type': 'line'},
             'mse': {'type': 'line'}},
         'meters':
            {'train_loss': {'type': 'AverageValueMeter', 'target': 'loss'},
             'test_loss': {'type': 'AverageValueMeter', 'target': 'loss'},
             'train_mse': {'type': 'MSEMeter', 'target': 'mse'},
             'test_mse': {'type': 'MSEMeter', 'target': 'mse'}
            }
         },
     'grads':
        {'plots':
            {'grad': {'type': 'line'}},
         'meters':
            {'grad': {'type': 'AverageValueMeter', 'target': 'grad'},
            }
         }
    }

_plot_defs = \
    {'simple':
        {'plots':
            {'loss': {'type': 'line'}},
         'meters':
            {'train_loss': {'type': 'AverageValueMeter', 'target': 'loss'},
             'test_loss': {'type': 'AverageValueMeter', 'target': 'loss'}}
         },
     'loss+MSE':
        {'plots':
            {'loss': {'type': 'line'},
             'mse': {'type': 'line'}},
         'meters':
            {'train_loss':
                 {'type': 'AverageValueMeter',
                  'target': 'loss'},
             'test_loss': {'type': 'AverageValueMeter', 'target': 'loss'},
             'train_mse': {'type': 'MSEMeter', 'target': 'mse',
                           'display': {'line': {'color': 'red',
                                                'dash': 'dash'}}
                           },
             'test_mse': {'type': 'MSEMeter', 'target': 'mse'}
            }
         },
     'loss+accuracy':
        {'plots':
            {'loss': {'type': 'line'},
             'accuracy': {'type': 'line'}},
         'meters':
            {'train_loss': {'type': 'AverageValueMeter', 'target': 'loss'},
             'test_loss': {'type': 'AverageValueMeter', 'target': 'loss'},
             'train_acc': {'type': 'AverageValueMeter', 'target': 'accuracy'},
             'test_acc':  {'type': 'AverageValueMeter', 'target': 'accuracy'}}
         },
     'grads':
        {'plots':
            {'grad': {'type': 'line'}},
         'meters':
            {'grad': {'type': 'AverageValueMeter', 'target': 'grad'},
            }
         }
    }


# PRESET CONFIGURATIONS
def get_preset_logger(key, **kwargs):
    if key in _plot_defs:
        cfg = _plot_defs[key]
        return FlexLogger(cfg['plots'], cfg['meters'], **kwargs)


def get_preset(key):
    cfg = _plot_defs.get(key, None)
    return cfg['plots'], cfg['meters']


def preset_names():
    return list(_plot_defs.keys())


# METERS AND PLOT INFO
def plot_types():
    return _meters


def meter_types():
    return _meters


def meter_info(name):
    return _meter_defs.get(name, None)


# Plots that require a model
def model_plot():
    pass


def fc_conv():
    pass










"""

meter + model + funcs -> layer_func_datasource_



"""












