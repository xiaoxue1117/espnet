from distutils.version import LooseVersion
import logging

import numpy as np
import six
import torch
import torch.nn.functional as F
import gtn
import math

from espnet.nets.pytorch_backend.nets_utils import to_device

class Allo(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alloG, alloW, log_probs, phone_arc_labels):
        B, T, C = log_probs.shape
        new_emissions_graphs = [None] * B
        emissions_graphs = [None] * B

        # create allophone graph
        #g_allophone = gtn.loadtxt(alloG)
        #g_allophone.set_weights(alloW.cpu().contiguous().data_ptr())
        #allophone_graphs = g_allophone

        def process(b):
            # create emission graph
            g_emissions = gtn.linear_graph(T, C, log_probs.requires_grad)
            cpu_data = log_probs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            g_new_emissions = gtn.intersect(g_emissions, alloG)
            emissions_graphs[b] = g_emissions
            new_emissions_graphs[b] = g_new_emissions

        gtn.parallel_for(process, range(B))
        #for b in range(B):
        #    process(b)

        ctx.auxiliary_data = (emissions_graphs, alloG, new_emissions_graphs, log_probs.shape, len(alloW), phone_arc_labels)

        # phoneme emissions
        new_emissions = torch.tensor([new_emissions_graphs[b].weights_to_list() for b in range(B)], \
                requires_grad=alloW.requires_grad, device=alloW.device).reshape(B, T, -1)

        return new_emissions.to(log_probs.device)

    @staticmethod
    def backward(ctx, grad_output):
        """backward
        """
        emissions_graphs, alloG, new_emissions_graphs, in_shape, out_shape, phone_arc_labels = ctx.auxiliary_data
        B, T, C = in_shape
        input_grad = torch.empty((B, T, C)) # log_probs

        def process(b):
            gtn.backward(new_emissions_graphs[b], False)
            input_grad[b] = torch.from_numpy(emissions_graphs[b]
                                                .grad()
                                                .weights_to_numpy()
                                            ).view(1, T, C)


        gtn.parallel_for(process, range(B))
        #for b in range(B):
        #    process(b)

        alloW_grad = torch.from_numpy(alloG.grad().weights_to_numpy()).to(grad_output.device)
        alloW_grad *= grad_output.sum(0).sum(0)

        grad_multiplier = []
        for phone in range(C):
            indices = [i for i, x in enumerate(phone_arc_labels) if x == phone]
            if len(indices) == 0:
                #grad_multiplier.append(torch.zeros((B, T), device=grad_output.device))
                grad_multiplier.append(torch.ones((B, T), device=grad_output.device))
            else:
                added = grad_output[:,:,indices]    # B x T x len(indices)
                added = torch.sum(added, dim=-1)    # B x T
                grad_multiplier.append(added)
        grad_multiplier = torch.stack(grad_multiplier, dim=-1)
        input_grad = input_grad.to(grad_output.device) * grad_multiplier / B

        #import pdb; pdb.set_trace()
        return (
            None,  # alloG
            alloW_grad,  # alloW
            input_grad,  # log_probs
            None,  #phone_arc_labels
        )

class AlloLayer(torch.nn.Module):
    """AlloLayer module
    """

    def __init__(self, allo_gtn, odim):
        super().__init__()
        self.alloG = gtn.loadtxt(allo_gtn)
        self.phone_arc_labels = self.alloG.labels_to_list()
        self.phoneme_arc_labels = gtn.project_output(self.alloG).labels_to_list()
        self.alloW = torch.tensor(self.alloG.weights_to_numpy())
        self.n_phonemes = odim
        self.fxn = Allo.apply

    def squash_many_phonemes_for_one_phone(self, new_emissions):
        new_emissions = torch.exp(new_emissions)    # return from log space
        B, T, _ = new_emissions.shape

        # add probs corresponding to the same phoneme
        squashed = []
        for phoneme in range(self.n_phonemes):
            indices = [i for i, x in enumerate(self.phoneme_arc_labels) if x == phoneme]
            added = new_emissions[:,:,indices]        # B x T x len(indices)
            added = torch.sum(added, dim=-1)          # B x T
            squashed.append(added)
        squashed = torch.stack(squashed, dim=-1)

        new_emissions = torch.nn.functional.log_softmax(squashed, dim=-1)
        return new_emissions

    def forward(self, hs_pad):
        """forward
        """
        log_probs = torch.nn.functional.log_softmax(hs_pad, dim=-1)

        new_emissions = self.fxn(self.alloG, self.alloW, log_probs, self.phone_arc_labels)
        new_emissions = self.squash_many_phonemes_for_one_phone(new_emissions)
        return new_emissions

