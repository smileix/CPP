import torch
import torch.nn as nn

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):


        self.batch_size = len(z_i)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        # 此处的zi与zj的shape是(bsz, project_dim), sim的shape是(2 * bsz, 2 * bsz),
        # todo sim矩阵是关于主对角线对称的，这两个相似矩阵也是相同的，
        sim = torch.matmul(z, z.T) / self.temperature
        # sim_i_j 与sim_j_i的shape都是(bsz,)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # cat之后的shape是(N,), reshape之后是(N,1)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # reshape之后是(N,N-2)
        negative_samples = sim[self.mask].reshape(N, -1)

        # 这里设置的其实可以看成是个N-1分类，bsz为N，判断该样本与其余N-1个样本中的哪个同属一个类别，
        # 每个batch两个view，可以看成共2 * bsz 个样本，每个样本都要与其他N-1个样本比较相似度，也就是N-1分类，只有同一个样本的不同view被认为是相似，否则不相似
        # todo 尝试，将这个实例级别的对比学习表示为若干个样本的二分类，具体来说是N * (N - 1) / 2 个样本的二分类
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N


        return loss


import numpy as np
from scipy.special import kl_div


def js_divergence(p, q):
    """Calculate the Jensen-Shannon Divergence between two probability distributions."""
    # Normalize the distributions so that they sum up to 1 along the last axis
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    m = 0.5 * (p + q)

    kl_p_m = p * (p.log() - m.log())
    kl_q_m = q * (q.log() - m.log())

    # Sum up the divergences along the last axis
    jsds = 0.5 * (kl_p_m.sum(dim=-1) + kl_q_m.sum(dim=-1))
    return jsds.mean()






