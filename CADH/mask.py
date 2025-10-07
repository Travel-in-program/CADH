import torch

from torch.nn.functional import interpolate as resize

#图像级掩码
'''def build_mask_generator(cfg):
    if cfg is None:
        return None
    t = cfg.pop('type')
    if t == 'block':
        return BlockMaskGenerator(**cfg)
    else:
        raise NotImplementedError(t)

class BlockMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs):
        print(imgs.shape)
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > self.mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask
'''
#特征级掩码
class FeatureMasker:
    """
    在特征上做随机掩码：
      - 向量 (B, D): 随机置零一部分维度，类似 feature-level dropout
      - 特征图 (B, C, H, W): 随机 block mask（下采样->随机->上采样）
    """
    def __init__(self, mask_ratio=0.5, block_size=8):
        self.mask_ratio = mask_ratio
        self.block_size = block_size

    @torch.no_grad()
    def mask_vec(self, feats_vec: torch.Tensor):
        # feats_vec: (B, D)
        B, D = feats_vec.shape
        keep = (torch.rand(B, D, device=feats_vec.device) > self.mask_ratio).float()
        return feats_vec * keep

    # @torch.no_grad()
    # def mask_map(self, feats_map: torch.Tensor):
    #     # feats_map: (B, C, H, W)
    #     print(feats_map)
    #     B, C, H, W = feats_map.shape
    #     h = max(1, round(H / self.block_size))
    #     w = max(1, round(W / self.block_size))
    #     mask_small = torch.rand(B, 1, h, w, device=feats_map.device)
    #     mask_small = (mask_small > self.mask_ratio).float()
    #     mask = F.interpolate(mask_small, size=(H, W), mode="nearest")
    #     return feats_map * mask
