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
    def forward(ctx, alloG, alloW, log_probs, phone_arc_labels, training=True):
        B, T, C = log_probs.shape
        new_emissions_graphs = [None] * B
        emissions_graphs = [None] * B

        # put weights into alloG since they will have been updated only in tensor form
        alloW_data = alloW.cpu().contiguous()
        alloG.set_weights(alloW_data.data_ptr())
        alloG.calc_grad = alloW.requires_grad
        alloG.zero_grad()

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
        if training:
            new_emissions = torch.tensor([new_emissions_graphs[b].weights_to_list() for b in range(B)], \
                    requires_grad=alloW.requires_grad, device=alloW.device).reshape(B, T, -1)
        else:
            new_emissions = torch.tensor([new_emissions_graphs[b].weights_to_list() for b in range(B)], \
                    requires_grad=False, device=alloW.device).reshape(B, T, -1)
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
        alloW_grad *= grad_output.sum(0).sum(0).to(alloW_grad.device)

        input_grad_mult = torch.zeros(input_grad.shape, device=grad_output.device, dtype=grad_output.dtype)
        input_grad_mult.index_add_(-1, phone_arc_labels.to(grad_output.device), grad_output)
        input_grad = input_grad.to(grad_output.device) * input_grad_mult

        return (
            None,  # alloG
            alloW_grad,  # alloW
            input_grad,  # log_probs
            None,  #phone_arc_labels
            None,  #training
        )

class AlloLayer(torch.nn.Module):
    """AlloLayer module
    """

    def __init__(self, allo_gtn, odim, trainable=True, redis=False, mask=None, lid=None, sm_allo=True, phoneme_bias=False):
        super().__init__()
        self.alloG = gtn.loadtxt(allo_gtn)
        self.phone_arc_labels = torch.tensor(self.alloG.labels_to_list())
        self.phoneme_arc_labels = torch.tensor(gtn.project_output(self.alloG).labels_to_list())

        if sm_allo:
            phones, counts = torch.unique(self.phone_arc_labels, return_counts=True)
            n_phones = len(phones)
            max_count = counts.max().item()
            self.alloWMask = torch.ones((n_phones, max_count)).bool()
            alloWDense = torch.ones((n_phones, max_count))
            # Make mask & init weights
            for p_idx in range(n_phones):
                self.alloWMask[p_idx, counts[p_idx]:] = False
                alloWDense[p_idx, counts[p_idx]:] = float('-inf')
            self.alloWDense = torch.nn.Parameter(alloWDense, requires_grad=True)
        else:
            if trainable:
                self.alloW = torch.nn.Parameter(torch.tensor(self.alloG.weights_to_numpy(), requires_grad=trainable))
            else:
                self.alloW = torch.tensor(self.alloG.weights_to_numpy(), requires_grad=trainable)

        self.n_phonemes = odim
        self.fxn = Allo.apply
        self.redis = redis
        self.mask = mask
        self.lid = lid
        self.sm_allo = sm_allo

        if phoneme_bias:
            self.phonemeW = torch.nn.Parameter(torch.zeros((self.n_phonemes), requires_grad=trainable)).unsqueeze(0).unsqueeze(0)

    def squash_many_phonemes_for_one_phone(self, new_emissions):
        new_emissions = new_emissions.exp()    # return from log space
        B, T, _ = new_emissions.shape

        # add probs corresponding to the same phoneme
        squashed_emissions = torch.zeros((B, T, self.n_phonemes), device=new_emissions.device, dtype=new_emissions.dtype)
        squashed_emissions.index_add_(-1, self.phoneme_arc_labels.to(new_emissions.device), new_emissions)
        # if the arcs are not dense, then this would not sum to 1
        squashed_emissions = squashed_emissions.log()
        return squashed_emissions

    def forward(self, hs_pad, training=True):
        """forward
        """
        if self.mask != None:
            hs_pad.masked_fill_(~self.mask.to(hs_pad.device), float('-inf'))
        log_probs = torch.nn.functional.log_softmax(hs_pad, dim=-1)

        if self.sm_allo:
            # normalize allo weights to be log probs
            alloW = self.alloWDense.log_softmax(dim=-1)[self.alloWMask==True]
            new_emissions = self.fxn(self.alloG, alloW, log_probs, self.phone_arc_labels)
        else:
            new_emissions = self.fxn(self.alloG, self.alloW, log_probs, self.phone_arc_labels, training)

        new_emissions = self.squash_many_phonemes_for_one_phone(new_emissions)

        if hasattr(self, 'phonemeW'):
            new_emissions = new_emissions + self.phonemeW.to(new_emissions.device)

        return new_emissions

    def get_alloW_SM(self):
        if hasattr(self, 'alloW'):
            phones, counts = torch.unique(self.phone_arc_labels, return_counts=True)
            max_count = counts.max().item()
            n_phones = len(phones)
            alloWMask = torch.ones((n_phones, max_count)).bool()
            alloWDense = torch.ones((n_phones, max_count))
            start = 0
            for p_idx in range(n_phones):
                alloWMask[p_idx, counts[p_idx]:] = False
                alloWDense[p_idx, :counts[p_idx]] = self.alloW[start: counts[p_idx]+start]
                alloWDense[p_idx, counts[p_idx]:] = float('-inf')
                start = start + counts[p_idx]
            alloWDenseSM = alloWDense.log_softmax(dim=-1)
            return alloWDenseSM[alloWMask==True]
        else:
            return self.alloWDense.log_softmax(dim=-1)[self.alloWMask==True]

def main():
    # 2 x 5 Emissions
    E = torch.rand((1, 2, 5), requires_grad=True).log_softmax(dim=-1)

    # graph, 5 arcs
    A = gtn.Graph()
    A.add_node(True, True)
    A.add_arc(0,0,0,0,math.log(1))  # <b>
    A.add_arc(0,0,1,1,math.log(1)) # a : A
    A.add_arc(0,0,2,1,math.log(1)) # b : A
    A.add_arc(0,0,3,2,math.log(.5)) # c : B
    A.add_arc(0,0,3,3,math.log(.5)) # c : C
    # phone d has no arc, or just redistribute
    #A.add_arc(0,0,4,1,math.log(1/3)) # d : A
    #A.add_arc(0,0,4,2,math.log(1/3)) # d : B
    #A.add_arc(0,0,4,3,math.log(1/3)) # d : C

    A_path = "/project/ocean/byan/espnet-ml/egs/babel/asr1/debug.gtn"
    gtn.savetxt(A_path, A)
    mask = torch.tensor([1, 1, 1, 1, 0]).unsqueeze(0).unsqueeze(0).bool()
    allolayer = AlloLayer(A_path, 4, mask=mask, sm_allo=False)

    print(E)
    #hs = allolayer(E)
    #from espnet.nets.pytorch_backend.ctc import CTC
    #ctcfxn = CTC(4, 4, 0.1, "builtin", reduce=True)
    #loss, _ = ctcfxn(hs, torch.tensor([2]), torch.tensor([[1, 2]]))
    #loss.backward()

    with torch.no_grad():
        hs_inference = allolayer(E, training=False)
        hs_inference = allolayer(E)

if __name__ == "__main__":
    main()
