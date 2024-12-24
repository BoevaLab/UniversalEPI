# Helper class for keeping track of training iterations
# adapted from https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/blob/master/Face_Enhancement/util/iter_counter.py

import os
import time

import numpy as np


class IterationCounter:
    def __init__(self, opt, dataset_size):
        self.opt = opt
        self.dataset_size = dataset_size
        self.first_epoch = 1
        self.total_epochs = opt.n_epochs
        self.epoch_iter = 0
        self.total_steps_so_far = (self.first_epoch - 1) * dataset_size + self.epoch_iter

        self.batchSize = None

    # Must reset the batch size when doing progression
    def set_batchsize(self, bs):
        self.batchSize = bs

    # return the iterator of epochs for the training
    def training_epochs(self):
        return range(self.first_epoch, self.total_epochs + 1)

    def record_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        self.epoch_iter = 0
        self.last_iter_time = time.time()
        self.current_epoch = epoch

    def record_one_iteration(self):
        current_time = time.time()

        self.time_per_iter = (current_time - self.last_iter_time) / self.batchSize
        self.last_iter_time = current_time
        self.total_steps_so_far += self.batchSize
        self.epoch_iter += self.batchSize

    def needs_printing(self):
        return (self.total_steps_so_far % self.opt.print_freq) < self.batchSize
