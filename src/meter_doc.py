import torch


_meter_defs = \
    {'MovingAverageValueMeter': [],
     'AUCMeter': [],
     'ClassErrorMeter': [],
     'MSEMeter': {'type': [torch.FloatTensor, torch.FloatTensor],
                  'spec': lambda x, y: x.size() == y.size(),
                  'doc': 'Two Tensors of same size'},
     'ConfusionMeter': [],
     'APMeter': [],
     'AverageValueMeter': {'type': float, 'doc': 'a single float or int'},
     'mAPMeter': []
     }