import numpy as np
import math

import torch.nn


class Upsample:
    def __init__(
            self,
            size=None,
            scale_factor=None,
            align_corners=None,
            recompute_scale_factor=None,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            self.size = size

        self.scale_factor = scale_factor
        self.align_corners = True if align_corners is True else False
        self.recompute_scale_factor = True if recompute_scale_factor is True else False

    def forward(self, input):
        h_out, w_out = None, None

        result = []

        for batch in input:
            channels, h_in, w_in = batch.shape

            if self.recompute_scale_factor or h_out is None or w_out is None:
                h_out, w_out = self.__compute_output_size(batch)

            out = np.zeros((channels, h_out, w_out), float)

            y_orig_center = (h_in - 1) / 2
            x_orig_center = (w_in - 1) / 2

            y_scaled_center = (h_out - 1) / 2
            x_scaled_center = (w_out - 1) / 2

            for c in range(channels):
                for y in range(h_out):
                    for x in range(w_out):
                        if self.align_corners:
                            x_ = max(1, w_in - 1) / max(1, w_out - 1) * x
                            y_ = max(1, h_in - 1) / max(1, h_out - 1) * y
                        else:
                            x_ = (x - x_scaled_center) / self.scale_factor + x_orig_center
                            y_ = (y - y_scaled_center) / self.scale_factor + y_orig_center

                        out[c, y, x] = self.bilinear_interpolation(batch, c, y_, x_)

            result.append(out)

        return result

    @staticmethod
    def bilinear_interpolation(batch, channel, y, x):
        height = batch.shape[1]
        width = batch.shape[2]

        x1 = max(min(math.floor(x), width - 1), 0)
        y1 = max(min(math.floor(y), height - 1), 0)
        x2 = max(min(math.ceil(x), width - 1), 0)
        y2 = max(min(math.ceil(y), height - 1), 0)

        a = float(batch[channel, y1, x1])
        b = float(batch[channel, y2, x1])
        c = float(batch[channel, y1, x2])
        d = float(batch[channel, y2, x2])

        dx = x - x1
        dy = y - y1

        new_pixel = a * (1 - dx) * (1 - dy)
        new_pixel += b * dy * (1 - dx)
        new_pixel += c * dx * (1 - dy)
        new_pixel += d * dx * dy

        return new_pixel

    def __compute_output_size(self, batch):
        if self.size is not None and self.scale_factor is not None:
            raise ValueError('Only one of size or scale_factor should be defined')
        elif self.size is None and self.scale_factor is None:
            raise ValueError('Size or scale factor should be specified!')
        elif self.size is not None and self.recompute_scale_factor:
            raise ValueError('recompute_scale_factor is not meaningful with an explicit size.')
        elif len(batch.shape) != 3:
            raise ValueError('Got {}D input, but bilinear mode needs 4D input'.format(len(batch.shape)))
        elif self.size is not None and len(self.size) == 2:
            return self.size[0], self.size[1]
        else:
            return math.floor(batch.shape[1] * self.scale_factor), \
                   math.floor(batch.shape[2] * self.scale_factor)
