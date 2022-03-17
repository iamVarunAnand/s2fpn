# import the necessary packages
from ..layers import MeshConv, MeshConvTransposeBilinear, DownSamp
from torch import nn
import torch

class ResBlock(nn.Module):
    def __init__(self, in_chan, neck_chan, out_chan, level, coarsen):
        # make a call to the parent constructor
        super(ResBlock, self).__init__()

        # get the path to the mesh file
        lvl = level - 1 if coarsen else level

        # initialise the instance variables
        self.coarsen = coarsen
        self.diff_chan = (in_chan != out_chan)

        # CONV 1x1 -> BN
        self.conv_1a = nn.Conv1d(in_chan, neck_chan, kernel_size=1, stride=1)
        self.bn_1a = nn.BatchNorm1d(neck_chan)

        # MESHCONV -> BN
        self.conv_2a = MeshConv(neck_chan, neck_chan, mesh_lvl=lvl, stride=1)
        self.bn_2a = nn.BatchNorm1d(neck_chan)

        # CONV 1x1 -> BN
        self.conv_3a = nn.Conv1d(neck_chan, out_chan, kernel_size=1, stride=1)
        self.bn_3a = nn.BatchNorm1d(out_chan)

        # RELU
        self.relu = nn.ReLU(inplace=True)

        # DOWNSAMPLE
        self.nv_prev = self.conv_2a.nv_prev
        self.down = DownSamp(self.nv_prev)

        # main branch
        if coarsen:
            self.seq_a = nn.Sequential(self.conv_1a, self.down, self.bn_1a, self.relu,
                                       self.conv_2a, self.bn_2a, self.relu,
                                       self.conv_3a, self.bn_3a)
        else:
            self.seq_a = nn.Sequential(self.conv_1a, self.bn_1a, self.relu,
                                       self.conv_2a, self.bn_2a, self.relu,
                                       self.conv_3a, self.bn_3a)

        # skip connection
        if self.diff_chan or coarsen:
            self.conv_1b = nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=1)
            # self.bn_1b = nn.BatchNorm1d(out_chan)
            self.bn_1b = nn.GroupNorm(32, out_chan)

            if coarsen:
                self.seq_b = nn.Sequential(self.conv_1b, self.down, self.bn_1b)
            else:
                self.seq_b = nn.Sequential(self.conv_1b, self.bn_1b)

    def forward(self, x):
        # skip connection
        if self.diff_chan or self.coarsen:
            x2 = self.seq_b(x)
        else:
            x2 = x

        # main branch
        x1 = self.seq_a(x)

        # addition
        out = x1 + x2

        # relu
        out = self.relu(out)

        # return the computation of the ResBlock
        return out


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, bias=True):
        """
            uses the mesh_file for the mesh of one-level up
        """

        # make a call to the parent constructor
        super(Up, self).__init__()

        # MESHCONV.T
        half_in = int(in_ch / 2)
        self.up = MeshConvTransposeBilinear(half_in, half_in, level, stride=2)

        # res block
        self.conv = ResBlock(in_ch, out_ch, out_ch, level, False)

    def forward(self, x1, x2):
        # upsample the previous pyramid level
        x1 = self.up(x1)

        # concatenate with features from encoder stage
        x = torch.cat([x2, x1], dim=1)

        # pass through res block
        x = self.conv(x)

        # return the layer computation
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, level, bias=True):
        """
            use the mesh_file for the mesh of one-level down
        """

        # make a call to the parent constructor
        super(Down, self).__init__()

        # res block
        self.conv = ResBlock(in_ch, in_ch, out_ch, level + 1, True)

    def forward(self, x):
        # pass the input through the res block and return
        return self.conv(x)


class SphericalUNet(nn.Module):
    def __init__(self, in_ch, out_ch, max_level=5, min_level=0, fdim=16):
        # make a call to the parent class constructor
        super(SphericalUNet, self).__init__()

        # initialise the instance variables
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level

        # initialise lists to store the encoder and decoder stages
        self.down, self.up = [], []

        # initial and final MESHCONV
        self.in_conv = MeshConv(in_ch, fdim, max_level, stride=1)
        self.out_conv = MeshConv(fdim, out_ch, max_level, stride=1)

        # encoder
        for i in range(self.levels - 1):
            # compute the number of in, out channels, and level
            ch_in = fdim * (2 ** i)
            ch_out = fdim * (2 ** (i + 1))
            lvl = max_level - i - 1

            # add a downsample block
            self.down.append(Down(ch_in, ch_out, lvl))

        # bottleneck
        ch_in = fdim * (2 ** (self.levels - 1))
        ch_out = fdim * (2 ** (self.levels - 1))
        self.down.append(Down(ch_in, ch_out, min_level))

        # decoder
        for i in range(self.levels - 1):
            # compute the number of in, out channels, and level
            ch_in = fdim * (2 ** (self.levels - i))
            ch_out = fdim * (2 ** (self.levels - i - 2))
            lvl = min_level + i + 1

            # add an upsample block
            self.up.append(Up(ch_in, ch_out, lvl))

        # final upsample
        self.up.append(Up(fdim * 2, fdim, max_level))

        # initialise the encoder and decoders as nn modules
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        # pass through initial MESHCONV
        x_ = [self.in_conv(x)]

        # loop through and pass the input through the encoder
        for i in range(self.levels):
            x_.append(self.down[i](x_[-1]))

        # print(f"[INFO] after down: {x_[-1].dtype}")

        # first upsample
        x = self.up[0](x_[-1], x_[-2])

        # print(f"[INFO] after first up: {x.dtype}")

        # loop through ans pass through the decoder
        for i in range(self.levels - 1):
            x = self.up[i + 1](x, x_[-3 - i])

        # print(f"[INFO] after up: {x.dtype}")

        # pass through final MESHCONV
        x = self.out_conv(x)

        # print(f"[INFO] after final conv: {x.dtype}")

        # return the output of the model
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = SphericalUNet(4, 15, max_level=5, fdim=32)

    summary(model, input_size=(1, 4, 10242))
