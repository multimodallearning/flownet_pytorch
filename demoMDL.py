import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.misc import imread
import scipy.io
import torch
from torch.autograd import Variable

from FlowNet2_src import FlowNet2
from FlowNet2_src import flow_to_image
from FlowNet2_src import *
import matplotlib.pyplot as plt

def overlaySegment(gray1,segs1):
    C, H, W = segs1.squeeze().size()
    colors=torch.FloatTensor([0,0,0,199,67,66,78,129,170,225,140,154,45,170,170,240,110,38,111,163,91,235,175,86,202,255,52,162,0,183]).view(-1,3)/255.0

    seg_color = torch.mm(segs1.view(3,-1).t(),colors[:3,:]).view(H,W,3)
    alpha = torch.clamp(1.0 - 0.5*(segs1[0,:,:]==0).float(),0,1.0)

    overlay = (gray1*alpha).unsqueeze(2) + seg_color*(1.0-alpha).unsqueeze(2)
    return overlay


if __name__ == '__main__':
  # Prepare img pair
#  im1 = imread('FlowNet2_src/example/0img0.ppm')
#  im2 = imread('FlowNet2_src/example/0img1.ppm')
  
  # Build model
  flownet2 = FlowNet2()
  path = 'FlowNet2_src/pretrained/FlowNet2_checkpoint.pth.tar'
  #path = 'FlowNet2_src/pretrained/FlowNet2-C_checkpoint.pth.tar'
  pretrained_dict = torch.load(path)['state_dict']
  model_dict = flownet2.state_dict()
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  flownet2.load_state_dict(model_dict)
  flownet2.cuda()
  test = scipy.io.loadmat('/data_supergrover2/heinrich/Uebungsdata/test_frames.mat')
  test_seg = scipy.io.loadmat('/data_supergrover2/heinrich/Uebungsdata/test_segmentatons.mat')
  test_seg1 = torch.from_numpy(test_seg['test_segmentations'])


  test_seg1 = torch.stack((test_seg1[:,:,:,1]==0,test_seg1[:,:,:,1]==50,test_seg1[:,:,:,1]==100),1).float()


  for i in range(23):
      im1 = test['test_frames'][i,:,:,:]
      im2 = test['test_frames'][i+1,:,:,:]
    
      seg1 = test_seg1[i,:,:,:]
      seg2 = test_seg1[i+1,:,:,:]
    
    #img = torch.zeros(1,3,256,320).cuda()
    #flow = torch.zeros(1,2,256,320).cuda()
    #warped = Resample2d()(img, flow)

      rgb0 = overlaySegment(torch.from_numpy(im1[:,:,1]).float()/255.0,seg2[:,:,:]).numpy()
      plt.imshow(rgb0)
      plt.savefig('noregMDL'+str(i)+'.png', bbox_inches='tight')


      # B x 3(RGB) x 2(pair) x H x W
      ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
      ims = torch.from_numpy(ims)
      print('i',i,'size',ims.size())
      ims_v = Variable(ims.cuda(), requires_grad=False)

        
      pred_flow = flownet2(ims_v)
      img_seg = seg2.unsqueeze(0).cuda()
      img_flow = pred_flow.data
      warped = Resample2d()(img_seg,img_flow)
      
      rgb1 = overlaySegment(torch.from_numpy(im1[:,:,1]).float()/255.0,warped.squeeze().data.cpu()).numpy()
      plt.imshow(rgb1)
      plt.savefig('overlayMDL'+str(i)+'.png', bbox_inches='tight')

      pred_flow = pred_flow[0].cpu().data.numpy().transpose((1,2,0))
      flow_im = flow_to_image(pred_flow)

      # Visualization
      plt.imshow(flow_im)
      plt.savefig('flowMDL'+str(i)+'.png', bbox_inches='tight')

      