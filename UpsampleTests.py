import unittest
from torch.nn import Upsample as TorchUpsample
from numpy.testing import assert_array_equal
import numpy as np
from Upsample import Upsample
from torch import from_numpy


class UpsampleTests(unittest.TestCase):
    def test_scale_factor_2(self):
        inputs = self.__get_data()

        own = Upsample(scale_factor=2)
        torch = TorchUpsample(scale_factor=2, mode='bilinear')

        for input in inputs:
            self.__assert(own, torch, input)

    def test_scale_factor_3(self):
        inputs = self.__get_data()

        own = Upsample(scale_factor=3)
        torch = TorchUpsample(scale_factor=3, mode='bilinear')

        for input in inputs:
            self.__assert(own, torch, input)

    def test_scale_factor_2_align_corners(self):
        inputs = self.__get_data()

        own = Upsample(scale_factor=2, align_corners=True)
        torch = TorchUpsample(scale_factor=2, align_corners=True, mode='bilinear')

        for input in inputs:
            self.__assert(own, torch, input)

    def test_scale_factor_3_align_corners(self):
        inputs = self.__get_data()

        own = Upsample(scale_factor=3, align_corners=True)
        torch = TorchUpsample(scale_factor=3, align_corners=True, mode='bilinear')

        for input in inputs:
            self.__assert(own, torch, input)

    def test_scale_factor_2_recompute(self):
        inputs = self.__get_data()

        own = Upsample(scale_factor=2, recompute_scale_factor=True)
        torch = TorchUpsample(scale_factor=2, recompute_scale_factor=True, mode='bilinear')

        for input in inputs:
            self.__assert(own, torch, input)

    def test_scale_factor_3_recompute(self):
        inputs = self.__get_data()

        own = Upsample(scale_factor=3, recompute_scale_factor=True)
        torch = TorchUpsample(scale_factor=3, recompute_scale_factor=True, mode='bilinear')

        for input in inputs:
            self.__assert(own, torch, input)

    @staticmethod
    def __assert(own: Upsample, torch: TorchUpsample, input):
        assert_array_equal(
            np.round(own.forward(input)),
            np.round(torch.forward(from_numpy(input)).detach().numpy())
        )

    @staticmethod
    def __get_data():
        return  np.array([[[[1., 2., 3.], [4., 5., 6.]]]]), \
                np.array([[[[17.]]]]), \
                np.array([[[[1., 1., 23.]]]]), \
                np.array([[[[1.], [2.], [3.]]]])


if __name__ == '__main__':
    unittest.main()
