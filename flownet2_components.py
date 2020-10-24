import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F

from flownet2_func_mph import * 

def conv(in_channels,
         out_channels,
         kernel_size=3,
         stride=1,
         bias=True,
         with_bn=True,
         with_relu=True):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True)
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if with_relu:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*tuple(layers))


def deconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True), nn.LeakyReLU(0.1, inplace=True))


def predict_flow(in_channels):
    return nn.Conv2d(
        in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)


class tofp16(nn.Module):

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):

    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i, j, :, :] = torch.from_numpy(bilinear)


def save_grad(grads, name):

    def hook(grad):
        grads[name] = grad

    return hook



'Parameter count = 581,226'

class FlowNetFusion(nn.Module):

    def __init__(self, with_bn=True):
        super(FlowNetFusion, self).__init__()

        self.with_bn = with_bn
        self.conv0 = conv(11, 64, with_bn=with_bn)
        self.conv1 = conv(64, 64, stride=2, with_bn=with_bn)
        self.conv1_1 = conv(64, 128, with_bn=with_bn)
        self.conv2 = conv(128, 128, stride=2, with_bn=with_bn)
        self.conv2_1 = conv(128, 128, with_bn=with_bn)

        self.deconv1 = deconv(128, 32)
        self.deconv0 = deconv(162, 16)

        self.inter_conv1 = conv(162, 32, with_bn=with_bn, with_relu=False)
        self.inter_conv0 = conv(82, 16, with_bn=with_bn, with_relu=False)

        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)

        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        flow2 = self.predict_flow2(out_conv2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((out_conv0, out_deconv0, flow1_up), 1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)

        return flow0

    'Parameter count , 39,175,298 '


class FlowNetC(nn.Module):

    def __init__(self, with_bn=True, fp16=False):
        super(FlowNetC, self).__init__()

        self.with_bn = with_bn
        self.fp16 = fp16

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
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)

        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)

        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)

        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,

        
        'Parameter count : 38,676,504 '


class FlowNetS(nn.Module):

    def __init__(self, input_channels=12, with_bn=True):
        super(FlowNetS, self).__init__()

        self.with_bn = with_bn
        self.conv1 = conv(
            input_channels, 64, kernel_size=7, stride=2, with_bn=with_bn)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv3_1 = conv(256, 256, with_bn=with_bn)
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
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,

'Parameter count = 45,371,666'


class FlowNetSD(nn.Module):

    def __init__(self, with_bn=True):
        super(FlowNetSD, self).__init__()

        self.with_bn = with_bn
        self.conv0 = conv(6, 64, with_bn=with_bn)
        self.conv1 = conv(64, 64, stride=2, with_bn=with_bn)
        self.conv1_1 = conv(64, 128, with_bn=with_bn)
        self.conv2 = conv(128, 128, stride=2, with_bn=with_bn)
        self.conv2_1 = conv(128, 128, with_bn=with_bn)
        self.conv3 = conv(128, 256, stride=2, with_bn=with_bn)
        self.conv3_1 = conv(256, 256, with_bn=with_bn)
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

        self.inter_conv5 = conv(1026, 512, with_bn=with_bn, with_relu=False)
        self.inter_conv4 = conv(770, 256, with_bn=with_bn, with_relu=False)
        self.inter_conv3 = conv(386, 128, with_bn=with_bn, with_relu=False)
        self.inter_conv2 = conv(194, 64, with_bn=with_bn, with_relu=False)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,
