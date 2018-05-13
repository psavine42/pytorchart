from src.logging import FlexLogger, _meters, _meter_defs


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


# PRESET CONFIGURATIONS
def get_preset_logger(key, **kwargs):
    if key in _plot_defs:
        cfg = _plot_defs[key]
        return FlexLogger(cfg['plots'], cfg['meters'], **kwargs)


def preset_info(key):
    return _plot_defs.get(key, None)


def preset_names():
    return list(_plot_defs.keys())


# METERS AND PLOT INFO
def plot_types():
    return _meters


def meter_types():
    return _meters


def meter_info(name):
    return _meter_defs.get(name, None)