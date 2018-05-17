

def get_in(o, kys, d=None):
    ob = o.copy()
    while ob and kys:
        k = kys.pop(0)
        ob = ob.get(k, None)
    if ob is None:
        return d
    return ob

