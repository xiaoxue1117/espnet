from distutils.version import LooseVersion
import logging

import numpy as np
import six
import torch
import torch.nn.functional as F
import gtn
import math

from espnet.nets.pytorch_backend.nets_utils import to_device

class AlloCTC(torch.autograd.Function):
    @staticmethod
    def create_ctc_graph(target, blank_idx):
        """Build gtn graph.
        :param list target: single target sequence
        :param int blank_idx: index of blank token
        :return: gtn graph of target sequence
        :rtype: gtn.Graph
        """
        g_criterion = gtn.Graph(False)
        L = len(target)
        S = 2 * L + 1
        for s in range(S):
            idx = (s - 1) // 2
            g_criterion.add_node(s == 0, s == S - 1 or s == S - 2)
            label = target[idx] if s % 2 else blank_idx
            g_criterion.add_arc(s, s, label)
            if s > 0:
                g_criterion.add_arc(s - 1, s, label)
            if s % 2 and s > 1 and label != target[idx - 1]:
                g_criterion.add_arc(s - 2, s, label)
        g_criterion.arc_sort(False)
        return g_criterion

    @staticmethod
    def forward(ctx, alloG, alloW, log_probs, targets, blank_idx=0):
        # TODO: log probs
        B, T, C = log_probs.shape
        new_emissions_graphs = [None] * B
        emissions_graphs = [None] * B
        allophone_graphs = [None] * B
        losses = [None] * B

        def process(b):
            # create emission graph
            g_emissions = gtn.linear_graph(T, C, log_probs.requires_grad)
            cpu_data = log_probs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            # create allophone graph
            g_allophone = gtn.loadtxt(alloG)
            g_allophone.set_weights(alloW.cpu().contiguous().data_ptr())

            # create criterion graph
            g_criterion = AlloCTC.create_ctc_graph(targets[b], blank_idx)

            # compose
            g_new_emissions = gtn.intersect(g_emissions, g_allophone)
            g_loss = gtn.negate(gtn.forward_score(gtn.intersect(g_new_emissions, g_criterion)))

            allophone_graphs[b] = g_allophone
            emissions_graphs[b] = g_emissions
            new_emissions_graphs[b] = g_new_emissions
            losses[b] = g_loss

        gtn.parallel_for(process, range(B))
        #for b in range(B):
        #    process(b)

        ctx.auxiliary_data = (emissions_graphs, allophone_graphs, new_emissions_graphs, losses, log_probs.shape, len(alloW))

        # phoneme emissions
        #new_emissions = torch.tensor([new_emissions_graphs[b].weights_to_list() for b in range(B)], \
        #        requires_grad=alloW.requires_grad, device=alloW.device).reshape(B, T, -1)
        loss = torch.tensor(\
                    [0.0 if (math.isnan(losses[b].item()) or math.isinf(losses[b].item())) \
                    else losses[b].item() for b in range(B)]\
                ).to(log_probs.device)
        loss = torch.mean(loss.cuda() if log_probs.is_cuda else loss)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """backward
        """
        emissions_graphs, allophone_graphs, new_emissions_graphs, losses, in_shape, out_shape = ctx.auxiliary_data
        B, T, C = in_shape
        input_grad = torch.empty((B, T, C)) # log_probs
        alloW_grad = torch.empty((B, out_shape)) #1D

        def process(b):
            gtn.backward(losses[b], False)
            i_grad = emissions_graphs[b].grad().weights_to_numpy()
            input_grad[b] = torch.from_numpy(i_grad).view(1, T, C)

            a_grad = allophone_graphs[b].grad().weights_to_numpy()
            alloW_grad[b] = torch.from_numpy(a_grad).view(1, out_shape)

        gtn.parallel_for(process, range(B))
        #for b in range(B):
        #    process(b)

        input_grad *= grad_output / B
        alloW_grad *= grad_output / B
        alloW_grad = torch.mean(alloW_grad, dim=0, keepdim=False)
        #import pdb;pdb.set_trace()

        return (
            None,  # alloG
            alloW_grad.to(grad_output.device),  # alloW
            input_grad.to(grad_output.device),  # log_probs
            None,  # targets
            None,  # blank_idx
        )

class AlloLayer(torch.nn.Module):
    """AlloLayer module
    """

    def __init__(self, allo_gtn):
        super().__init__()
        self.alloG = allo_gtn
        #alloW = torch.tensor(gtn.loadtxt(self.alloG).weights_to_numpy())
        #self.alloW = torch.nn.Parameter(alloW, requires_grad=True) #torch.zeros(10)#allo_W
        self.fxn = AlloCTC.apply
        self.ignore_id = -1

    def forward(self, alloW, hs_pad, hlens, ys_pad):
        """forward
        """
        ys = [y[y != self.ignore_id] for y in ys_pad]
        targets = [t.tolist() for t in ys]

        log_probs = torch.nn.functional.log_softmax(hs_pad, dim=2)

        loss = self.fxn(self.alloG, alloW, log_probs, targets, 0)
        return loss

