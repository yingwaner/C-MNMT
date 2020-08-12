# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, batchsize, timestep, loss_threshold, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    #compute each sentences loss and continue the too difficult sentence
    nll_loss_tmp = nll_loss.view(batchsize, timestep)
    non_pad_mask = target.ne(ignore_index).view(batchsize, timestep)
    nll_loss_tmp = nll_loss_tmp.mul(non_pad_mask.float()).sum(1)
    target_tmp = target.clone().view(batchsize, timestep)
    process_num = 0
    #for i in range(batchsize):
    #    if nll_loss_tmp[i] > loss_threshold:
    #        target_tmp[i] = ignore_index
    #        continue
    #    process_num = process_num + 1
    target_tmp = target_tmp.view(-1, 1)

    if ignore_index is not None:
        non_pad_mask = target_tmp.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss, process_num, batchsize


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, lang_pair, loss_threshold, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(lang_pair, **sample['net_input'])
        loss, nll_loss, process_num, total_num = self.compute_loss(model, net_output, sample, loss_threshold, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output, process_num, total_num

    def compute_loss(self, model, net_output, sample, loss_threshold, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        [batchsize, timestep, _] = list(lprobs.size())
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss, process_num, total_num = label_smoothed_nll_loss(
            lprobs, target, self.eps, batchsize, timestep, loss_threshold, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss, process_num, total_num

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
