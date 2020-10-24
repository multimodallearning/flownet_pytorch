# flownet_pytorch
converting Nvidia's pytorch FlowNet with only builtin layers to support to newer pytorch versions

 this implementation heavily draws from the following publication, please cite this if you use any of the layers:
 
 @inproceedings{heinrich2019closing,
  title={Closing the gap between deep and conventional image registration using probabilistic dense displacement networks},
  author={Heinrich, Mattias P},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={50--58},
  year={2019},
  organization={Springer}
}
https://arxiv.org/pdf/1907.10931
 
 pytorch-flownet with NVIDIA code is not supported in current pytorch versions
 due to depreciated torch.utils.ffi, no fix available (https://github.com/pytorch/pytorch/issues/15645)
 here a re-implementation of all external functions Resample2d, ChannelNorm and Correlation
 with built-in pytorch modules is provided, (requires v1.0+)
 this also enables running flownet on cpu and further improvements (e.g. fine-tuning with segmentations)

 for original code base for model definitions, see https://github.com/NVIDIA/flownet2-pytorch
 and https://github.com/vt-vl-lab/pytorch_flownet2
 additionally, flowlib.py by Ruoteng Li could be handy

you need some utility files and pretrained models from https://github.com/vt-vl-lab/flownet2.pytorch

