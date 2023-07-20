import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class AllNegativeTripletSelector():
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, symmetric=False, cpu=False):
        super(AllNegativeTripletSelector, self).__init__()
        self.symmetric = symmetric
        self.cpu = cpu

    def get_triplets(self, anchors, positives):
        if self.cpu:
            anchors = anchors.cpu()
        anchor_num = anchors.shape[0]
        shape_num = positives.shape[0]
        triplets = []
        shapes = np.array([i for i in range(shape_num)])
        if self.symmetric:
            for i in range(anchor_num):
                negative_indices = np.where(shapes != i)[0]
                triplets.extend([[i, i + anchor_num, negative + anchor_num] for negative in negative_indices])
                triplets.extend([[i + anchor_num, i, negative] for negative in negative_indices])
        else:
            for i in range(anchor_num):
                negative_indices = np.where(shapes != i)[0]
                triplets.extend([[i, negative] for negative in negative_indices])

        triplets = np.array(triplets)
        return torch.LongTensor(triplets)

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector, beta=1):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.softplus = nn.Softplus(beta=beta)

    def forward(self, anchors,  positives, softplus=False):
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        triplets = self.triplet_selector.get_triplets(anchors,  positives)
        if anchors.is_cuda:
            triplets = triplets.cuda()
        ap_distances = (anchors[triplets[:, 0]] - positives[triplets[:, 0]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (anchors[triplets[:, 0]] - positives[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        if softplus:
            losses = self.softplus(ap_distances - an_distances + self.margin)
        else:
            losses = F.relu(ap_distances - an_distances + self.margin)
        return losses.mean()