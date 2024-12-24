# MIT License
# Copyright (c) 2020 Yutaro Yamada
# https://github.com/runopti/stg

import math

import torch
import torch.nn as nn


class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01 * torch.randn(input_dim), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.device = device

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate

        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        """Gaussian CDF."""
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self
