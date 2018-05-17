

_def_opts = {'markersize': 10,
             'colormap': 'Viridis',
             'markersymbol': 'dot',
             'markers': False,
             'fillarea': False}

_def_layout = {'layout': {'margin': {'t': 60, 'r': 60, 'b': 60, 'l': 60},
               'showlegend': True}}

_spec = \
    {'dash': ['dash', 'dot', 'dashdot'],
     'connectgaps': [True, False],
     'type': ['scatter'],
     'name': str,
     'mode': ['lines', 'markers', 'lines+markers'],
     'line.color': str,
     'line.width': int,
     'line.shape': ["linear", "spline","hv", "vh", "hvh","vhv"],
     'line.dash': ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"],
     'marker.size': int,
     'marker.symbol': ['dot'],
     'marker.color': str,
     'marker.line.width': float,
     'marker.line.color': str,
     }