# -*- coding: utf-8 -*-
# pylint: disable=protected-access
from __future__ import absolute_import
import logging

import torch
from torch.nn import functional as F


LOGGER = logging.getLogger(__name__)


# based : https://github.com/pytorch/pytorch/blob/15e8bb379ef06a1a88b31b5f0dad843ee32b26df/torch/nn/functional.py#L1193-L1250
# added beta parameter
def _gumbel_softmax_sample(logits, tau=1., beta=1., eps=1e-10, noise=None):
    # type: (Tensor, float, float, float, Tensor) -> Tensor
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    if noise is None or noise.size(0) == 0:
        gumbel_noise = F._sample_gumbel(logits.size(), eps=eps, out=torch.empty_like(logits))
    else:
        gumbel_noise = noise
    y = logits + (beta * gumbel_noise)
    return F.softmax(y / tau, dims - 1), gumbel_noise

def gumbel_softmax(logits, tau=1., beta=1., hard=False, eps=1e-10, noise=None):
    # type: (Tensor, float, float, bool, float, Tensor) -> Tensor
    r"""
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: `[batch_size, num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
    Returns:
      Sampled tensor of shape ``batch_size x num_features`` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across features
    Constraints:
    - Currently only work on 2D input :attr:`logits` tensor of shape ``batch_size x num_features``
    Based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft, new_noise = _gumbel_softmax_sample(logits, tau=tau, beta=beta, eps=eps, noise=noise)
    _ = new_noise
    if hard:
        _, k = y_soft.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(shape, dtype=logits.dtype, device=logits.device).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y


def sampling_gumbel_softmax(logits, num_samples, tau=1., beta=1., hard=False, eps=1e-10):
    shape = logits.size()
    assert len(shape) == 2
    y_soft = gumbel_softmax(logits, tau=tau, beta=beta, hard=False, eps=eps)

    sampled_index = torch.multinomial(y_soft, num_samples, replacement=False)
    y_hard = torch.zeros(shape, dtype=logits.dtype, device=logits.device).scatter_(-1, sampled_index.view(1, -1), 1.0)
    if hard:
        y = y_hard - y_soft.detach() + y_soft
    else:
        #y_hard = y_hard - y_soft.detach() + y_soft
        y_weighed = y_hard * y_soft
        y_normalized = y_weighed / y_weighed.norm(p=1)
        y = y_normalized.detach() - y_soft.detach() + y_soft
        #y = y_normalized
    return y
