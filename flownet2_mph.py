# by Mattias Heinrich 
# pytorch-flownet with NVIDIA code is not supported in current pytorch versions
# due to depreciated torch.utils.ffi, no fix available (https://github.com/pytorch/pytorch/issues/15645)
# here a re-implementation of all external functions Resample2d, ChannelNorm and Correlation
# with built-in pytorch modules is provided, (requires v1.0+)
# this also enables running flownet on cpu and further improvements (e.g. fine-tuning with segmentations)

# for original code base for model definitions, see https://github.com/NVIDIA/flownet2-pytorch
# and https://github.com/vt-vl-lab/pytorch_flownet2
# additionally, flowlib.py by Ruoteng Li could be handy

import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F



from flownet2_func_mph import * 

from flownet2_components import * 
# this includes FlowNetC, FlowNetS, FlowNetSD, FlowNetFusion
# and conv, deconv, predict_flow tofp16, tofp32, save_grad



class FlowNet2(nn.Module):

    def __init__(self,
                 with_bn=False,
                 fp16=False,
                 rgb_max=255.,
                 div_flow=20.,
                 grads=None):
        super(FlowNet2, self).__init__()
        self.with_bn = with_bn
        self.div_flow = div_flow
        self.rgb_max = rgb_max
        self.grads = {} if grads is None else grads

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC(with_bn=with_bn, fp16=fp16)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample1 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS(with_bn=with_bn)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample2 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS(with_bn=with_bn)

        # Block (FlowNetSD)
        self.flownets_d = FlowNetSD(with_bn=with_bn)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')

        self.resample3 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())
        self.resample4 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())

        # Block (FLowNetFusion)
        self.flownetfusion = FlowNetFusion(with_bn=with_bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1, )).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1, ))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            [x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0],
            dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat(
            (x, resampled_img1, flownets1_flow / self.div_flow,
             norm_diff_img0),
            dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)

        diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        req_grad = diff_flownets2_flow.requires_grad
        if req_grad:
            diff_flownets2_flow.register_hook(
                save_grad(self.grads, 'diff_flownets2_flow'))

        diff_flownets2_img1 = self.channelnorm(
            (x[:, :3, :, :] - diff_flownets2_flow))
        if req_grad:
            diff_flownets2_img1.register_hook(
                save_grad(self.grads, 'diff_flownets2_img1'))

        # flownetsd
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)

        diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        if req_grad:
            diff_flownetsd_flow.register_hook(
                save_grad(self.grads, 'diff_flownetsd_flow'))

        diff_flownetsd_img1 = self.channelnorm(
            (x[:, :3, :, :] - diff_flownetsd_flow))
        if req_grad:
            diff_flownetsd_img1.register_hook(
                save_grad(self.grads, 'diff_flownetsd_img1'))

        # concat img1 flownetsd, flownets2, norm_flownetsd, norm_flownets2,
        # diff_flownetsd_img1, diff_flownets2_img1
        concat3 = torch.cat(
            (x[:, :3, :, :], flownetsd_flow, flownets2_flow,
             norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1,
             diff_flownets2_img1),
            dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)

        if req_grad:
            flownetfusion_flow.register_hook(
                save_grad(self.grads, 'flownetfusion_flow'))

        return flownetfusion_flow


#class FlowNet2C(FlowNetC):
#
#    def __init__(self, with_bn=False, fp16=False, rgb_max=255., div_flow=20):
#        super(FlowNet2C, self).__init__(with_bn, fp16)
#        self.rgb_max = rgb_max
#        self.div_flow = div_flow
#
#    def forward(self, inputs):
#        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1, )).mean(
#            dim=-1).view(inputs.size()[:2] + (1, 1, 1, ))
#
#        x = (inputs - rgb_mean) / self.rgb_max
#        x1 = x[:, :, 0, :, :]
#        x2 = x[:, :, 1, :, :]#
#
#        flows = super(FlowNet2C, self).forward(x1, x2)
#
#        if self.training:
#            return flows
#        else:
#            return self.upsample1(flows[0] * self.div_flow)

class FlowNet2C(nn.Module):
    def __init__(self, rgb_max=255, batchNorm=False, div_flow=20):
        super(FlowNet2C,self).__init__()
        self.rgb_max = rgb_max
        self.div_flow = div_flow
        with_bn = batchNorm
        fp16 = False

        self.conv1 = conv(3, 64, kernel_size=7, stride=2, with_bn=with_bn)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv_redir = conv(
            256, 32, kernel_size=1, stride=1, with_bn=with_bn)

        corr = Correlation(
            pad_size=20,
            kernel_size=1,
            max_displacement=20,
            stride1=1,
            stride2=2,
            corr_multiply=1)
        self.corr = nn.Sequential(tofp32(), corr, tofp16()) if fp16 else corr

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(473, 256, with_bn=with_bn)
        self.conv4 = conv(256, 512, stride=2, with_bn=with_bn)
        self.conv4_1 = conv(512, 512, with_bn=with_bn)
        self.conv5 = conv(512, 512, stride=2, with_bn=with_bn)
        self.conv5_1 = conv(512, 512, with_bn=with_bn)
        self.conv6 = conv(512, 1024, stride=2, with_bn=with_bn)
        self.conv6_1 = conv(1024, 1024, with_bn=with_bn)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=True)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.uniform(m.bias)
                nn.init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.uniform(m.bias)
                nn.init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]

        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)
        
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b) # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)

        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)

        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1,out_deconv3,flow4_up),1)

        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2S(FlowNetS):

    def __init__(self, with_bn=False, rgb_max=255., div_flow=20):
        super(FlowNet2S, self).__init__(input_channels=6, with_bn=with_bn)
        self.rgb_max = rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1, )).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1, ))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)

        flows = super(FlowNet2S, self).forward(x)

        if self.training:
            return flows
        else:
            return self.upsample1(flows[0] * self.div_flow)


class FlowNet2SD(FlowNetSD):

    def __init__(self, with_bn=False, rgb_max=255., div_flow=20):
        super(FlowNet2SD, self).__init__(with_bn=with_bn)
        self.rgb_max = rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1, )).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1, ))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)

        flows = super(FlowNet2SD, self).forward(x)

        if self.training:
            return flows
        else:
            return self.upsample1(flows[0] * self.div_flow)


class FlowNet2CS(nn.Module):

    def __init__(self, with_bn=False, fp16=False, rgb_max=255., div_flow=20):
        super(FlowNet2CS, self).__init__()
        self.with_bn = with_bn
        self.fp16 = fp16
        self.rgb_max = rgb_max
        self.div_flow = div_flow

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC(with_bn=with_bn, fp16=fp16)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.resample1 = (nn.Sequential(tofp32(), Resample2d(), tofp16())
                          if fp16 else Resample2d())

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS(with_bn=with_bn)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1, )).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1, ))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            [x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0],
            dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        return flownets1_flow


class FlowNet2CSS(nn.Module):

    def __init__(self, with_bn=False, fp16=False, rgb_max=255., div_flow=20):
        super(FlowNet2CSS, self).__init__()
        self.with_bn = with_bn
        self.fp16 = fp16
        self.rgb_max = rgb_max
        self.div_flow = div_flow

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC(with_bn=with_bn, fp16=fp16)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS(with_bn=with_bn)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if fp16:
            self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample2 = Resample2d()

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS(with_bn=with_bn)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1, )).mean(
            dim=-1).view(inputs.size()[:2] + (1, 1, 1, ))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            [x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0],
            dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat(
            (x, resampled_img1, flownets1_flow / self.div_flow,
             norm_diff_img0),
            dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)

        return flownets2_flow
