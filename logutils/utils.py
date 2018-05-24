from copy import deepcopy
from functools import reduce


def get_in(o, kys, d=None):
    ob = o.copy()
    while ob and kys:
        k = kys.pop(0)
        ob = ob.get(k, None)
    if ob is None:
        return d
    return ob


def deep_merge(*dicts, update=False):
    """
    Merges dicts deeply.
    Parameters
    ----------
    dicts : list[dict]
        List of dicts.
    update : bool
        Whether to update the first dict or create a new dict.
    Returns
    -------
    merged : dict
        Merged dict.

    shamelesly lifted from  https://gist.github.com/yatsu/68660bea18edfe7e023656c250661086
    """
    def merge_into(d1, d2):
        for key in d2:
            if key not in d1 or not isinstance(d1[key], dict):
                d1[key] = deepcopy(d2[key])
            else:
                d1[key] = merge_into(d1[key], d2[key])
        return d1

    if update:
        return reduce(merge_into, dicts[1:], dicts[0])
    else:
        return reduce(merge_into, dicts, {})


