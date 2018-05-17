from visdom import Visdom
from .style_utils import _def_opts, _def_layout, _spec
import pickle, pprint


class TraceLogger(object):
    def __init__(self,
                 *args,
                 opts={},
                 vis=None,
                 title=[],
                 env=None,
                 port=8097,
                 **kwargs):
        self._title = title
        self._port  = port
        self._win   = None
        self._env   = env if env is not None else 'main'
        self._opts  = {**_def_opts, **opts.get('opts', {})}
        self._layout = {**_def_layout, **opts.get('layout', {})}
        self._lines  = self.init_lines(title, opts.get('data', {}))
        self._viz    = vis if isinstance(vis, Visdom) else Visdom(port=port)

    @property
    def viz(self):
        return self._viz

    def save(self, path):
        pickle.dump(self, path)
        return self._viz.save([self._env])

    @staticmethod
    def init_lines(titles, opts):
        """

        :param titles:
        :param opts:
        :return:

        otps usage :
             {'line1':{
                'name': '1',
                'type': 'scatter',
                  'marker': {'size': 10,
                             'symbol': 'dot',
                             'line': {'width': 0.5, 'color': '#000000'}},
                  'mode': 'lines'}}
        """
        def check_trace(opts, pre=''):
            _opts = {}
            for k, value in opts.items():
                fk = k if pre == '' else pre + '.' + k
                if isinstance(value, dict):
                    _opts[k] = check_trace(value, fk)
                else:
                    spec = _spec.get(fk, None)
                    if isinstance(spec, list) and value in spec:
                        _opts[k] = value
                    elif isinstance(spec, type) and isinstance(value, spec):
                        _opts[k] = value
            return _opts
        # print('==========')
        # print(titles)
        # pprint.pprint(opts)
        lines = []
        for title in titles:
            trace_style = opts.get(title, {})
            opts_dict = check_trace(trace_style)
            # todo required keys
            opts_dict['type'] = 'scatter'
            opts_dict['mode'] = 'lines'
            lines.append(opts_dict)
        # pprint.pprint(lines)
        return lines

    def _create_trace(self, X, Y):
        data_to_send = {
            'data': [],
            'win': self._win,
            'eid': self._env,
            'layout': self._layout,
            'opts': self._opts,
        }
        assert len(X) == len(Y), 'X and Y inputs not same size'
        for i, (x, y) in enumerate(zip(X, Y)):
            line_dict = self._lines[i].copy()
            line_dict['x'] = [x]
            line_dict['y'] = [y]
            data_to_send['data'].append(line_dict)
        return data_to_send

    def log(self, X, Y):
        print(X, Y)
        ds = self._create_trace(X, Y)
        if self._win is not None:
            ds['append'] = True
            ds['win'] = self._win
            self._viz._send(ds, endpoint='update')
        else:
            print('starting plot ')
            self._win = self._viz._send(ds, endpoint='events')

    def __repr__(self):
        st = '\n'
        for spec in self._lines:
            st += _unwrap(spec) + '\n'
        return st


def _unwrap(dict_, pre=''):
    st = ''
    for k, v in dict_.items():
        if isinstance(v, dict):
            st += pre + k + ': \n' + _unwrap(v, pre + ' '*3)
        else:
            st += pre + k + ': ' + str(v) + '\n'
    return st


