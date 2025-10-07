import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import torch.nn.functional as F
#from mask import mask_vec_blockwise

def classification_loss(pred_src, labels_src, pred_tgt, pseudo_labels_tgt):
    #交叉熵公式
    
    return F.cross_entropy(pred_src, labels_src.long())+ F.cross_entropy(pred_tgt, pseudo_labels_tgt.long())#.argmax(axis=1)   1.5*F.cross_entropy(pred_tgt, pseudo_labels_tgt.long())
'''
def domain_loss(pred, domain_labels):
    return F.cross_entropy(pred, domain_labels)'''

def hash_pairwise_loss(hash_relaxed_src, hash_relaxed_tgt, sim_matrix, alpha=0.5):
    # 两个不同的hash码
    Hs = F.normalize(hash_relaxed_src)
    #print(Hs)
    Ht = F.normalize(hash_relaxed_tgt)
    
    cos_sim =Hs.mm(Ht.t())  # (Bs, Bt)
    eps = 1e-9
    sim_loss = torch.mean(torch.pow(cos_sim - sim_matrix, 2))
    reg_loss = torch.mean(torch.sum(torch.pow(torch.sign(hash_relaxed_tgt+eps) - hash_relaxed_tgt, 2), dim=1))
    #print(sim_loss,reg_loss)reg_loss损失值大
    #print(torch.sign(hash_relaxed_tgt))
    loss = (1-alpha) * sim_loss + alpha * reg_loss
    return loss

def quantization_loss(relaxed_hash):
    # encourage relaxed hash to be near -1 or 1
    return (relaxed_hash.abs() - 1.0).pow(2).mean()


'''def mask_consistency_loss(model, class_logits_target,x_target, mask_generator, num_classes):
    # ========== 1. 完整图像预测 ==========求解w
    logits_full = class_logits_target
    probs_full = F.softmax(logits_full, dim=1)
    pseudo_labels = probs_full.argmax(dim=1)
    conf_weights = probs_full.max(dim=1)[0]
    # ========== 2. 掩码图像预测 ==========
    x_masked = mask_generator.mask_vec(x_target)
    out_masked = model.predict(x_masked)#module.
    logits_masked = out_masked['class_logits']
    # ========== 3. 一致性损失 ==========
    loss_mask = -(conf_weights * torch.sum(
        F.one_hot(pseudo_labels, num_classes=num_classes).float().to(x_target.device) *
        F.log_softmax(logits_masked, dim=1), dim=1)
    ).mean()

    return loss_mask'''
def mask_consistency_loss(model, class_logits_target,x_target, mask_generator,pseudo_labels1, num_classes):
    # ========== 1. 完整图像预测 ==========求解w
    logits_full = class_logits_target
    probs_full = F.softmax(logits_full, dim=1)
    pseudo_labels = probs_full.argmax(dim=1)
    conf_weights = probs_full.max(dim=1)[0]
    
    #print('conf_weights:',conf_weights)
    # ========== 2. 掩码图像预测 ==========

    x_masked = mask_generator.mask_vec(x_target)
    out_masked = model.predict(x_masked)#module.
    logits_masked = out_masked['class_logits']
    # print('logits_masked:',logits_masked)
    # print('pseudo_labels:',pseudo_labels)
    # print('pseudo_labels1:',pseudo_labels1)
    
    # ========== 3. 一致性损失 ==========
    loss_per_sample = F.cross_entropy(logits_masked, pseudo_labels1, reduction="none")
    loss_mask = (conf_weights * loss_per_sample).mean()

    return loss_mask

#该部分源域标签与目标域标签都为one—hot编码的标签，多分类器，
def domain_classification_loss(pred_src, labels_src, pred_tgt, pseudo_labels_tgt, num_classes):
    #标签进行one-hot编码

    #y_src_onehot = F.one_hot(labels_src, num_classes=num_classes).float()
    #y_tgt_onehot = pseudo_labels_tgt.float() 

    loss_src = F.cross_entropy(pred_src, labels_src.long())
    loss_tgt = F.cross_entropy(pred_tgt, pseudo_labels_tgt.long())
    return loss_src + loss_tgt
    #return 0.4*loss_src + 0.6*loss_tgt


#域对齐损失
def domain_adversarial_loss(pred_src, pred_tgt):
    device = pred_src.device
    #domain_criterion = torch.nn.CrossEntropyLoss()
#源域是0
    src_labels = torch.zeros(pred_src.size(0), dtype=torch.long, device=device)  
# 目标域标签全 1
    tgt_labels = torch.ones(pred_tgt.size(0), dtype=torch.long, device=device)  
    domain_criterion = torch.nn.NLLLoss()
    loss_src =domain_criterion(pred_src, src_labels)
    loss_tgt =domain_criterion(pred_tgt, tgt_labels)
    #loss_src = domain_criterion(pred_src, src_labels)
    #loss_tgt = domain_criterion(pred_tgt, tgt_labels)
    #print(loss_src,loss_tgt)
    loss_domain = loss_src + loss_tgt
    #loss_domain = 0.3*loss_src + 0.7*loss_tgt
    return loss_domain
class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss