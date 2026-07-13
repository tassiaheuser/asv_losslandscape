import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy


# # Tassis Chat GPT Version
# class LossFunction(nn.Module):
#     def __init__(self, scale=30.0, margin=0.35, **kwargs):
#         super(LossFunction, self).__init__()
#         self.scale = scale
#         self.margin = margin

#     def forward(self, logits, labels):
#         cosine = F.linear(F.normalize(logits), F.normalize(logits.weight))
#         one_hot = torch.zeros(cosine.size(), device=logits.device)
#         one_hot.scatter_(1, labels.view(-1, 1), 1)
#         logits_with_margin = self.scale * (cosine - one_hot * self.margin)
#         loss = F.cross_entropy(logits_with_margin, labels)
#         return loss


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.m,self.s))

    def forward(self, x, label=None):

        # assert x.size()[0] == label.size()[0]
        # assert x.size()[2] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        # costh = torch.mm(x_norm, w_norm) # only works for 2D matrices
        costh = x_norm @ w_norm
        label_view = label.view(-1, 1)
        # label_view = label.repeat(x.shape[1],1)[None].permute(2,1,0).cpu()
        # if label_view.is_cuda: label_view = label_view.cpu() # not needed, bad code
        # delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        delt_costh = torch.zeros((costh.shape[0],costh.shape[2]),device=label_view.device).scatter_(1, label_view, self.m)

        # if x.is_cuda: delt_costh = delt_costh.to(x.device)

        delt_costh = delt_costh.repeat(x.shape[1], 1, 1).permute(1, 0, 2)
    
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m

        label_utterances_batched = label.repeat(costh_m_s.shape[1], 1).T.reshape(-1)
        costh_batched = costh_m_s.reshape(-1,costh_m_s.shape[-1])
        # loss    = self.ce(costh_m_s.reshape(-1,costh_m_s.shape[0]), label)
        loss    = self.ce(costh_batched, label_utterances_batched)
        # prec1   = accuracy(costh_m_s.detach(), label_utterances.detach(), topk=(1,))[0]
        
        prec1   = accuracy(costh_batched.detach(), label_utterances_batched.detach(), topk=(1,))[0]
        return loss, prec1

