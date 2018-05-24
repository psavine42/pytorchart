import torch
from torchvision.utils import make_grid
from torchvision.transforms import Resize


class ImageMeter(object):
    def __init__(self, size=50, nrow=4):
        super(ImageMeter, self).__init__()
        self.val = []
        self.cnt = 0
        self._nrow = nrow
        self._nimg = nrow ** 2
        self._size = size
        self._xform = Resize(size)
        self.reset()

    def add(self, image):
        if self.cnt > self._nimg:
            return
        _size = list(image.size())
        if image.dim() == 4:
            size = _size[1:]
            self.cnt += _size[0]
        elif image.dim() == 3:
            size = _size
            image.unsqueeze(0)
            self.cnt += 1
        elif image.dim() == 2:
            size = list(image.size())
            size.insert(0, 1)
            image.unsqueeze(0).unsqueeze(0)
            self.cnt += 1
        else:
            return
        if size != self._size:
            image = self._xform(image)
        self.val.append(image)

    def value(self):
        return make_grid(torch.cat(self.val), nrow=self._nrow)

    def reset(self):
        self.val = []
        self.cnt = 0

