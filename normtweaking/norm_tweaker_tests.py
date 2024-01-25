import unittest
import torch
from normtweaking import NormTweaker, LinearQuantizer

class TestNormTweaker(unittest.TestCase):
    def test_channelwise_distribution_loss_equal(self):
        tweaker = NormTweaker(None, None)

        fout = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        qout = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        loss = tweaker._channelwise_distribution_loss(fout, qout)
        self.assertEqual(loss, 0)

    def test_channelwise_distribution_loss_correct_dim(self):
        tweaker = NormTweaker(None, None)

        fout = torch.tensor([[[1, 2, 3],
                              [4, 5, 6]]], dtype=torch.float32)
        qout = torch.tensor([[[1, 9, 3],
                              [8, 5, 6]]], dtype=torch.float32)
        loss = tweaker._channelwise_distribution_loss(fout, qout)
        # means:   [2.5,  3.5, 4.5]
        #          [4.5,  7.0, 4.5]
        # vars :   [4.5,  4.5, 4.5]
        #          [24.5, 8.0, 4.5]
        # ||μ||  : [2.0,  3.5, 0.0]
        # ||σ^2||: [20.0, 3.5, 0.0]
        # sum: 2.0 + 3.5 + 0.0 + 20.0 + 3.5 + 0.0 = 29.0
        # loss: 29.0 / 3 = 9.66666
        self.assertAlmostEqual(loss.item(), 9.6666, delta=0.0001)


if __name__ == '__main__':
    unittest.main()