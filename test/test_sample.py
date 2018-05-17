import torch
import pprint, random
import unittest
from visdom import Visdom
from logutils import get_preset_logger, FlexLogger, TraceLogger
import numpy as np


des = {'win': None,
       'opts': {'markersize': 10,
                'colormap': 'Viridis',
                'mode': 'lines',
                'markers': False,
                'fillarea': False,
                'markersymbol': 'dot'},
       'data': [{'type': 'scatter',
                 'y': [4, 5, 6],
                 'mode': 'lines',
                 'marker': {'size': 10,
                            'symbol': 'dot',
                            'line': {'width': 0.5, 'color': '#000000'}},
                 'x': [1, 2, 3], 'name': '1'}],
       'eid': None,
       'layout': {'showlegend': False,
                  'margin': {'r': 60, 't': 60, 'l': 60, 'b': 60}}}


des2 = {'opts': {'markersize': 10,
                 'colormap': 'Viridis',
                 'markersymbol': 'dot',
                 'markers': False,
                 'fillarea': False,
                 'mode': 'lines'},
        'data': [{'name': '1',
                  'type': 'scatter',
                  'marker': {'size': 10,
                             'symbol': 'dot',
                             'color': 'red',
                             'line': {'width': 0.5, 'color': '#000000'}},
                  'mode': 'markers+lines',
                  'line': {'color': 'red',
                           'dash':'dash'},
                  },
                 {'name': '2',
                  'type': 'scatter',
                  'marker': {'size': 10,
                             'symbol': 'dot',
                             'line': {'width': 0.5, 'color': '#000000'}},
                  'mode': 'lines'}],
        'layout': {'margin': {'t': 60, 'r': 60, 'b': 60, 'l': 60}, 'showlegend': False},
        'win': None,
        'eid': None}

upd = {'layout': {'margin': {'l': 60, 't': 60, 'r': 60, 'b': 60},   'showlegend': False},
       'append': True,
       'name': None,
       'data': [{'y': [1.5], 'type': 'scatter', 'x': [2.0], 'name': '1',
                 'marker': {'symbol': 'dot', 'line': {'width': 0.5, 'color': '#000000'},
                            'size': 10}, 'mode': 'lines'},
                {'y': [1.6],
                 'type': 'scatter', 'x': [2.0], 'name': '2',
                 'marker': {'symbol': 'dot',
                            'line': {'width': 0.5, 'color': '#000000'},
                            'size': 10},
                 'mode': 'lines'}],
       'win': 'window_363b9f162e3d1e',
       'eid': None,
       'opts': {'fillarea': False,
                'colormap': 'Viridis',
                'markersymbol': 'dot',
                'markers': False,
                'markersize': 10,
                'mode': 'lines'}}


class TestSample(unittest.TestCase):
    def test_traces(self):
        titles = ['1', '2']
        data = {o['name']: o for o in des2['data']}
        opts = {}
        opts['data'] = data
        tlg = TraceLogger(win='test', title=titles, opts=opts)
        pprint.pprint(tlg._lines)
        tlg.log([3, 3], [5, 6])
        tlg.log([4, 4], [4, 5])

        # print(str(tlg))
        # pprint.pprint(res)

    def test_opts(self):
        titles = ['1', '2']
        opts = {o['name']: o for o in  des2['data']}

        lines = TraceLogger.init_lines(titles, opts)
        pprint.pprint(lines)


    def test_baseline(self):
        v = Visdom()
        w = v.line(X=np.array([[1, 1]]), Y=np.array([[1, 2]]))
        v.line(X=np.array([[2, 2]]), Y=np.array([[1.5, 1.6]]), win=w, update='append')
