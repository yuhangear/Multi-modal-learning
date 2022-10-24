import torch
from torch import nn
import numpy as np
import pdb

def sequence_mask(sequence_length, max_len=None, device=None):
    #pdb.set_trace()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = torch.LongTensor(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    #seq_range_expand = seq_range_expand.to(device)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()



class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="sum")
        self.criterion2 = nn.CosineEmbeddingLoss(margin=0.2)

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        
        mask_ = mask.expand_as(input)
        loss = self.criterion2((input * mask_).reshape(-1,256), (target * mask_).reshape(-1,256),torch.ones(input.shape[0]*input.shape[1]).to(input.device))
        #return loss / mask.sum()
        return loss 


class MaskedMSELoss2(nn.Module):
    def __init__(self):
        super(MaskedMSELoss2, self).__init__()
        self.criterion = nn.MSELoss(reduction="sum")

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        #loss = nn.functional.cross_entropy(input * mask_, target * mask_)
        return loss / mask.sum()
        #return loss


class MaskedCELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="sum")
       

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        
        mask_ = mask.expand_as(input)
        #loss = self.criterion((input * mask_).reshape(-1,256), (target * mask_).reshape(-1,256),torch.ones(input.shape[0]*input.shape[1]).to(input.device))
        loss = self.criterion((input * mask_), (target * mask_))
        return loss / mask.sum()
        return loss 

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction="sum")

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        #return loss / mask.sum()
        return loss / mask_.sum()


if __name__ == "__main__":
    seq_lengths = torch.LongTensor([1,3, 12, 5])
    mask = sequence_mask(seq_lengths, max_len=15)
    print(mask)
