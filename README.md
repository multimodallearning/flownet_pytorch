# flownet_pytorch
converting Nvidia's pytorch FlowNet with only builtin layers to support to newer pytorch versions

 pytorch-flownet with NVIDIA code is not supported in current pytorch versions
 due to depreciated torch.utils.ffi, no fix available (https://github.com/pytorch/pytorch/issues/15645)
 here a re-implementation of all external functions Resample2d, ChannelNorm and Correlation
 with built-in pytorch modules is provided, (requires v1.0+)
 this also enables running flownet on cpu and further improvements (e.g. fine-tuning with segmentations)

 for original code base for model definitions, see https://github.com/NVIDIA/flownet2-pytorch
 and https://github.com/vt-vl-lab/pytorch_flownet2
 additionally, flowlib.py by Ruoteng Li could be handy

you need some utility files and pretrained models from https://github.com/vt-vl-lab/flownet2.pytorch

