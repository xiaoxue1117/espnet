from distutils.version import LooseVersion
import logging

import numpy as np
import six
import torch
import torch.nn.functional as F
import gtn
import math

from espnet.nets.pytorch_backend.nets_utils import to_device

class AlloBiphoneCTC(torch.autograd.Function):

    @staticmethod
    def create_ctc_graph(target, blank_idx=0):
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
    def forward(ctx, alloG, alloW, biphoneG, biphoneW, log_probs, targets):
        B, T, C = log_probs.shape
        emissions_graphs = [None] * B
        losses = [None] * B

        # put weights into alloG since they will have been updated only in tensor form
        alloW_data = alloW.cpu().contiguous()
        alloG.set_weights(alloW_data.data_ptr())
        alloG.calc_grad = alloW.requires_grad
        alloG.zero_grad()

        biphoneW_data = biphoneW.cpu().contiguous()
        biphoneG.set_weights(biphoneW_data.data_ptr())
        biphoneG.calc_grad = biphoneW.requires_grad
        biphoneG.zero_grad()

        biphoneAlloG = gtn.intersect(biphoneG, alloG)

        def process(b):
            # create emission graph
            g_emissions = gtn.linear_graph(T, C, log_probs.requires_grad)
            cpu_data = log_probs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            g_new_emissions = gtn.intersect(g_emissions, biphoneAlloG)
            ## biphone transitions
            #g_biphone_emissions = gtn.intersect(g_emissions, biphoneG)

            #g_new_emissions = gtn.intersect(g_biphone_emissions, alloG)

            # create criterion graph
            g_criterion = AlloBiphoneCTC.create_ctc_graph(targets[b])
            g_loss = gtn.negate(gtn.forward_score(gtn.intersect(g_new_emissions, g_criterion)))

            emissions_graphs[b] = g_emissions
            losses[b] = g_loss

        gtn.parallel_for(process, range(B))
        #for b in range(B):
        #    process(b)

        #ctx.auxiliary_data = (emissions_graphs, alloG, biphoneG, biphoneAlloG, losses, log_probs.shape, len(alloW))
        ctx.auxiliary_data = (emissions_graphs, alloG, biphoneG, losses, log_probs.shape, len(alloW))

        #loss = torch.tensor([losses[b].item() for b in range(B)]).to(log_probs.device)
        loss = torch.tensor([losses[b].item() for b in range(B) if not math.isinf(losses[b].item())]).to(log_probs.device)
        loss = torch.mean(loss.cuda() if log_probs.is_cuda else loss)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """backward
        """
        emissions_graphs, alloG, biphoneG, losses, in_shape, out_shape = ctx.auxiliary_data
        #emissions_graphs, alloG, biphoneG, biphoneAlloG, losses, in_shape, out_shape = ctx.auxiliary_data
        B, T, C = in_shape
        input_grad = torch.empty((B, T, C)) # log_probs

        def process(b):
            gtn.backward(losses[b], True)
            #gtn.backward(losses[b], False)
            input_grad[b] = torch.from_numpy(emissions_graphs[b]
                                                .grad()
                                                .weights_to_numpy()
                                            ).view(1, T, C)

        gtn.parallel_for(process, range(B))
        #for b in range(B):
        #    process(b)
        alloW_grad = torch.from_numpy(alloG.grad().weights_to_numpy()).to(grad_output.device)
        #alloW_grad *= grad_output / B
        alloW_grad *= grad_output

        biphoneW_grad = torch.from_numpy(biphoneG.grad().weights_to_numpy()).to(grad_output.device)
        biphoneW_grad *= grad_output

        input_grad = input_grad.to(grad_output.device)
        input_grad *= grad_output / B
        return (
            None,  # alloG
            alloW_grad,  # alloW
            None,  # biphoneG
            biphoneW_grad,  # biphoneW
            input_grad,  # log_probs
            None,  #targets
        )

def make_transitions_graph(ngram, num_tokens, calc_grad=False):
    import itertools
    transitions = gtn.Graph(calc_grad)
    transitions.add_node(True, ngram == 1)

    state_map = {(): 0}

    # first build transitions which include <s>:
    for n in range(1, ngram):
        for state in itertools.product(range(num_tokens), repeat=n):
            in_idx = state_map[state[:-1]]
            out_idx = transitions.add_node(False, ngram == 1)
            state_map[state] = out_idx
            transitions.add_arc(in_idx, out_idx, state[-1])

    for state in itertools.product(range(num_tokens), repeat=ngram):
        state_idx = state_map[state[:-1]]
        new_state_idx = state_map[state[1:]]
        # p(state[-1] | state[:-1])
        transitions.add_arc(state_idx, new_state_idx, state[-1])

    if ngram > 1:
        # build transitions which include </s>:
        end_idx = transitions.add_node(False, True)
        for in_idx in range(end_idx):
            transitions.add_arc(in_idx, end_idx, gtn.epsilon)

    return transitions

class AlloBiCTCLayer(torch.nn.Module):
    """AlloLayer module
    """

    def __init__(self, allo_gtn, odim, idim, trainable=True, redis=False, mask=None, lid=None, sm_allo=True, phoneme_bias=False):
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
        self.fxn = AlloBiphoneCTC.apply
        self.redis = redis
        self.mask = mask
        self.lid = lid
        self.sm_allo = sm_allo

        self.biphoneG = make_transitions_graph(2, idim, True)
        self.biphoneW = torch.nn.Parameter(torch.tensor(self.biphoneG.weights_to_numpy(), requires_grad=trainable))

    def forward(self, hs_pad, targets, training=True):
        """forward
        """
        # GTN CTC expects unpadded list; ignore_id = -1
        targets = [y[y != -1] for y in targets]

        if self.mask != None:
            hs_pad = hs_pad.masked_fill(~self.mask.to(hs_pad.device), float('-inf'))
        log_probs = torch.nn.functional.log_softmax(hs_pad, dim=-1)

        if self.sm_allo:
            # normalize allo weights to be log probs
            alloW = self.alloWDense.log_softmax(dim=-1)[self.alloWMask==True]
            loss = self.fxn(self.alloG, alloW, self.biphoneG, self.biphoneW, log_probs, targets)
        else:
            loss = self.fxn(self.alloG, self.alloW, self.biphoneG, self.biphoneW, log_probs, targets)

        return loss

    def predict(self, hs_pad):
        """forward
        """
        if self.mask != None:
            hs_pad = hs_pad.masked_fill(~self.mask.to(hs_pad.device), float('-inf'))
        log_probs = torch.nn.functional.log_softmax(hs_pad, dim=-1)

        if self.sm_allo:
            # normalize allo weights to be log probs
            alloW = self.alloWDense.log_softmax(dim=-1)[self.alloWMask==True]
        else:
            alloW = self.alloW

        #forward pass with no grad
        B, T, C = log_probs.shape
        new_emissions_graphs = [None] * B

        # put weights into alloG since they will have been updated only in tensor form
        alloW_data = alloW.cpu().contiguous()
        alloG = self.alloG
        alloG.set_weights(alloW_data.data_ptr())
        alloG.calc_grad = False

        def process(b):
            # create emission graph
            g_emissions = gtn.linear_graph(T, C, log_probs.requires_grad)
            cpu_data = log_probs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            g_new_emissions = gtn.intersect(g_emissions, alloG)

            new_emissions_graphs[b] = g_new_emissions

        gtn.parallel_for(process, range(B))

        new_emissions = torch.tensor([new_emissions_graphs[b].weights_to_list() for b in range(B)], \
                requires_grad=False, device=alloW.device).reshape(B, T, -1)

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
    torch.manual_seed(10)
    E = torch.rand((2, 3, 5), requires_grad=True).log_softmax(dim=-1)
    #E = torch.tensor([[[-1.8498, -1.3670, -1.8210, -1.5794, -1.5146],
    #     [-2.0829, -1.4438, -1.5144, -1.2768, -1.9624]]])

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
    allolayer = AlloBiCTCLayer(A_path, 4, mask=mask, sm_allo=False)

    print(E)
    targets = torch.tensor([[1,2,3],[1,2,3]])
    loss = allolayer(E, targets)
    print(loss)
    loss.backward()
    print(allolayer.alloW.grad)
    print(allolayer.biphoneW.grad)

    #hs = allolayer.predict(E)
    #print(hs)

if __name__ == "__main__":
    main()
