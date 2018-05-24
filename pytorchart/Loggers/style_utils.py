"""

Default Definitions and specifications for simple type checking for plot configs

def_{} are defaults used to setup a minimal plot

spec_{} are options that I am slowly pulling out of the plotly docs.
this is somewhat an exersise in understanding plotly and documenting.
on the other hand, it may be useful to have around.

todo - should i generate these with a metaprogram? probably.

"""


_def_opts = \
    {'markersize': 10,
     'colormap': 'Viridis',
     'markersymbol': 'dot',
     'markers': False,
     'connectgaps': True,
     'fillarea': False}


_def_layout = \
    { # 'layout': {'margin': {'t': 60, 'r': 60, 'b': 60, 'l': 60}, ]
     # 'yaxis': {'type':'log', 'autorange':True,},
                #'tickvals':[.0001, .001, .01, .1, 1, 10]},
     # 'width':300,
     # 'height':300,
     'showlegend': True}

lyout_spec = \
    {'showlegend': [True, False],
     'xaxis.type': ['log'],
     'yaxis.type': ['log']
    }


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
     'line.smoothing': float,   # range (0 and 1.3)
     'line.simplify': [True, False],

     'marker.size': int,
     'marker.symbol': ['dot'],
     'marker.color': str,
     'marker.line.width': float,
     'marker.line.color': str,
     }



