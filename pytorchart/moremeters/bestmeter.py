

class BestValueMeter(object):
    """
    Keep track of best value for a metric.
    Compares metrics and discards wrose value

    :options:
      -if top=True, then max(new, current) is kept
      -if top=False, then min(new, current) is kept


    """
    def __init__(self, metric=None, top=True):
        self.reset()
        self._top = top
        self._metric = metric
        self.m = 0
        self.val = 0

    def add(self, value, m=None):
        if self._metric is not None and m is not None:
            mv = self._metric(m)
            if self._top is True and mv > self.m:
                self.val = value
                self.m = mv
            elif self._top is False and mv < self.m:
                self.val = value
                self.m = mv
        elif self._top is True and value > self.val:
            self.val = value
        elif self._top is False and value < self.val:
            self.val = value

    def value(self):
        return self.val

    def reset(self):
        self.val = 0
        self.m = 0

