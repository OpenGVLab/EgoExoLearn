import torch
from torch.autograd import Variable

class RankingAttentionLoss(torch.nn.Module):
    def __init__(self, margin=0.1):
        super(RankingAttentionLoss,self).__init__()
        self.att_margin = margin
        
    def forward(self, input1_1, input1_2, input2_1, input2_2):
        _losses1 = input1_1.clone()
        _losses1.add_(-1, input1_2)
        
        _losses2 = input2_1.clone()
        _losses2.add_(-1, input2_2)
        
        _losses = _losses2.clone()
        _losses.add_(-1, _losses1)
        _losses.add_(self.att_margin)
        _losses.clamp_(min=0)
        loss = _losses.sum()/(input1_1.size(0))
        return loss
        
