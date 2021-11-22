# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import logging
from torch.utils.data import sampler
from utils import AverageMeter

class BatchSampler(sampler.Sampler):
    """
    Simply determines the order in which the loader will read samples from the data set.
    We want to sample batches randomly, but each batch should have samples that are
    close to each other in the dataset (so that we don't have a lot of zero padding)
    """

    def __init__(self, dataset, batch_size, randomize=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.randomize=randomize

        batches = [range(begin_id, begin_id + batch_size) for begin_id in range(0, len(dataset), batch_size)]

        # if the indexes in the last batch are going over len(dataset), we drop the last batch.
        if batches[-1][-1] > len(dataset):
            if drop_last:
                del batches[-1]
            else:
                batches[-1]=range(batches[-1][0],len(dataset))
        self.batches = batches

    def __iter__(self):

        if self.randomize:
            random.shuffle(self.batches)

        return iter(self.batches)

    def __len__(self):
        return len(self.batches) * self.batch_size


class DynamicBatchSampler(sampler.Sampler):
    """Extension of Sampler that will do the following:
        1.  Change the batch size (essentially number of sequences)
            in a batch to ensure that the total number of frames are less
            than a certain threshold.
        2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(self, sampler, frames_threshold, max_batch_size=0, unsorted_batch=False, fps= 1000 / 30):
        """
        @sampler: will mostly be an instance of DistributedSampler.
        Though it should work with any sampler.
        @frames_threshold: maximum area of the batch
        """
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_batch_size = max_batch_size
        self.unsorted_batch = unsorted_batch

        indices, batches = list(), list()
        # the dataset to which these indices are pointing to
        dataset = self.sampler.dataset
        # get all the indices and corresponding durations from
        # the sampler
        for idx in self.sampler:
            indices.append((idx, dataset.utt_list[idx]["duration"]))

        # sort the indices according to duration
        if self.unsorted_batch is False:
            indices.sort(key=lambda elem : elem[1])
            max_dur = indices[-1][1]
        else:
            # make sure that you will be able to serve all the utterances
            max_dur = max([indices[i][1] for i in range(len(indices))])

        # start clubbing the utterances together
        batch = list()
        batch_frames, batch_area = 0, 0
        max_frames_in_batch = 0
        average_meter = AverageMeter('Padding Efficiency')
        for idx, duration in indices:
            if duration > 0:
                frames = duration * fps
                if frames > max_frames_in_batch:
                    max_frames_in_batch = frames

                if (self.unsorted_batch and len(batch) < max_batch_size)\
                    or (not self.unsorted_batch and batch_frames + frames <= self.frames_threshold and (max_batch_size == 0 or len(batch) < max_batch_size)):
                    batch.append(idx)
                    batch_frames += frames
                    batch_area = max_frames_in_batch * len(batch)
                else:
                    # log the stats and add previous batch to batches
                    if batch_area > 0 and len(batch) > 0:
                        average_meter.add(batch_frames, batch_area)
                        batches.append(batch)
                    # make a new one
                    batch = list()
                    batch_frames, batch_area = frames, frames
                    max_frames_in_batch = batch_frames

        # When all indices are processed
        if batch_area > 0 and len(batch) > 0:
            average_meter.add(batch_frames, batch_area)
            batches.append(batch)

        # don't need the 'indices' any more
        del indices
        self.batches = batches
        average_meter.display_results(loglevel=logging.DEBUG)

    def __iter__(self):
        # shuffle on a batch level
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

