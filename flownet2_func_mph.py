# by Mattias Heinrich 
# pytorch-flownet with NVIDIA code is not supported in current pytorch versions
# due to depreciated torch.utils.ffi, no fix available (https://github.com/pytorch/pytorch/issues/15645)
# here a re-implementation of all external functions Resample2d, ChannelNorm and Correlation
# with built-in pytorch modules is provided, (requires v1.0+)
# this also enables running flownet on cpu and further improvements (e.g. fine-tuning with segmentations)
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F

#own re-implementation (mattias)
def Correlation(pad_size,kernel_size,max_displacement,stride1,stride2,corr_multiply):
    #todo throw error when unexpected input combinations occur
    disp_hw = max_displacement
    corr_unfold = torch.nn.Unfold((disp_hw+1,disp_hw+1),dilation=(stride2,stride2),padding=disp_hw)

    def applyCorr(feat1,feat2):
        B,C,H,W = feat1.size()
        return torch.mean(corr_unfold(feat2).view(B,C,-1,H,W)*(feat1).unsqueeze(2),1)

    return applyCorr
#own re-implementation (mattias)
def ChannelNorm():
    def applyChNorm(A):
        return torch.sqrt(torch.sum(A**2,1,keepdim=True))
    return applyChNorm
#own re-implementation (mattias)
def Resample2d():    
    def applyResample(img,pix_flow):
        #expects pix_flow to have size B x 2x H x W
        B,C,H,W = img.size()
        disp_flow = pix_flow.to(img.device)/(torch.Tensor([W,H]).to(img.device)-1).view(1,-1,1,1)*2
        identity = F.affine_grid(torch.eye(2,3).unsqueeze(0).repeat(B,1,1),torch.Size((B,1,H,W))).to(img.device)
        deformation = (identity+disp_flow.permute(0,2,3,1))
        warped = F.grid_sample(img,deformation)
        return warped
    return applyResample