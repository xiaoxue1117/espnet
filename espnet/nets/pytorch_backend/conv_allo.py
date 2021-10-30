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
        num_arcs = len(phone_arc_labels)
        alloW_grad = torch.from_numpy(alloG.grad().weights_to_numpy()).view(1, T, num_arcs).to(grad_output.device)
        alloW_grad = alloW_grad.expand(B,-1,-1) / B
        alloW_grad *= grad_output.to(alloW_grad.device)

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

class ConvAlloLayer(torch.nn.Module):
    """AlloLayer module
    """

    def __init__(self, allo_gtn, idim, n_phones, odim, mask=None, lid=None, kernel=5, use_layer_norm=False):
        super().__init__()
        self.alloG = gtn.loadtxt(allo_gtn)
        self.alloG.calc_grad = False    #this is only used for construction of T x AlloG
        self.phone_arc_labels = torch.tensor(self.alloG.labels_to_list())
        self.phoneme_arc_labels = torch.tensor(gtn.project_output(self.alloG).labels_to_list())
        self.num_arcs = len(self.phone_arc_labels)

        phones, counts = torch.unique(self.phone_arc_labels, return_counts=True)
        unique_phones = len(phones)
        max_count = counts.max().item()
        self.alloWMask = torch.ones((unique_phones, max_count)).bool()
        # Make mask & init weights
        for p_idx in range(unique_phones):
            self.alloWMask[p_idx, counts[p_idx]:] = False

        self.n_phones = n_phones
        self.n_phonemes = odim
        self.fxn = Allo.apply
        self.mask = mask
        self.lid = lid

        self.kernel=kernel
        self.use_layer_norm=use_layer_norm

        #conv1d
        padding = (kernel-1) // 2    #same padding
        self.conv = torch.nn.Conv2d(1, idim, (kernel, idim), padding=(padding, 0))
        #out to alloWDense, which then can be masked
        self.W_out = torch.nn.Sequential(
            torch.nn.Linear(idim, idim),
            torch.nn.Dropout(.1),
            torch.nn.Linear(idim, unique_phones * max_count)
        )
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(odim)

    def squash_many_phonemes_for_one_phone(self, new_emissions):
        new_emissions = new_emissions.exp()    # return from log space
        B, T, _ = new_emissions.shape

        # add probs corresponding to the same phoneme
        squashed_emissions = torch.zeros((B, T, self.n_phonemes), device=new_emissions.device, dtype=new_emissions.dtype)
        new_emissions = squashed_emissions.index_add(-1, self.phoneme_arc_labels.to(new_emissions.device), new_emissions)
        #squashed_emissions.index_add_(-1, self.phoneme_arc_labels.to(new_emissions.device), new_emissions)
        # if the arcs are not dense, then this would not sum to 1
        new_emissions = new_emissions.log()
        return new_emissions

    def make_expand_graph(self, T):
        V = self.n_phones
        allo = gtn.Graph(False)
        allo.add_node(True, False)
        for t in range(1, T):
          allo.add_node(False, False)
        allo.add_node(False, True)
        for t in range(T):
          for v in range(V):
            allo.add_arc(t,t+1,v,v)
        return allo

    def forward(self, phone_out, hs_pad, training=True):
        """forward
        phone_out is the log_softmax of phone output layer
        hs_pad is hidden states pre phone output layer
        """
        hs_pad = hs_pad.unsqueeze(1)  # b x 1 x t x i
        hs_pad = self.conv(hs_pad)    # b x o x t x 1
        hs_pad = hs_pad.squeeze(-1).transpose(1,2)    # b x t x o
        if self.use_layer_norm:
            hs_pad = self.layer_norm(hs_pad)
        #BxTxnum_arcs
        W = self.W_out(hs_pad)
        B, T, _ = W.shape

        #need expand graph to make AlloG over T
        expandG = self.make_expand_graph(W.shape[1])
        alloG = gtn.intersect(expandG, self.alloG)

        if self.mask != None:
            phone_out = phone_out.masked_fill(~self.mask.to(phone_out.device), float('-inf'))
        log_probs = torch.nn.functional.log_softmax(phone_out, dim=-1)

        #need to expand allowmask to make it over BxT
        unique_phones, max_count = self.alloWMask.shape
        alloWMask = self.alloWMask.to(W.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        W = W.view(B, T, unique_phones, max_count).masked_fill(~alloWMask, float('-inf'))
        alloW = W.log_softmax(dim=-1)[alloWMask==True].view(B, T, -1)

        new_emissions = self.fxn(alloG, alloW, log_probs, self.phone_arc_labels, training)
        new_emissions = self.squash_many_phonemes_for_one_phone(new_emissions)

        return new_emissions

def main():
    # 2 x 5 Emissions
    torch.manual_seed(10)
    H = torch.rand((2, 20, 10), requires_grad=True)
    E = torch.rand((2, 20, 5), requires_grad=True).log_softmax(dim=-1)
    #E = torch.ones((1, 2, 5), requires_grad=True).log_softmax(dim=-1)

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
    allolayer = ConvAlloLayer(A_path, idim=10, odim=4, mask=mask, kernel=3)

    print(E)
    hs = allolayer(E, H)
    from espnet.nets.pytorch_backend.ctc import CTC
    #ctcfxn = CTC(4, 4, 0.1, "gtnctc", reduce=True)
    ctcfxn = CTC(4, 4, 0.1, "builtin", reduce=True)
    loss, _ = ctcfxn(hs, torch.tensor([3,3]), torch.tensor([[1,2,3],[1,2,3]]))
    print(loss)
    loss.backward()
    print(allolayer.alloW.grad)
    #with torch.no_grad():
    #    hs_inference = allolayer(E, training=False)
    #    hs_inference = allolayer(E)

if __name__ == "__main__":
    main()
