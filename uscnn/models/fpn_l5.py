# import the necessary packages
from ..layers import MeshConv, ResBlock, MeshConvTranspose, MeshConvTransposeBilinear
from torch import nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, up=MeshConvTranspose):
        """
            use mesh_file for the mesh of one-level up
        """

        # make a call to the parent constructor
        super(Up, self).__init__()

        # MESHCONV.T
        self.up = up(out_ch, out_ch, level, stride=2)
        self.up_bn = nn.BatchNorm1d(out_ch)
        self.up_relu = nn.ReLU(inplace=True)

        # cross connection
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1)
        self.cross_bn = nn.BatchNorm1d(out_ch)
        self.cross_relu = nn.ReLU(inplace=True)

        # final meshconv
        self.out_conv = MeshConv(out_ch, out_ch, level)
        self.out_bn = nn.BatchNorm1d(out_ch)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        # upsample the previous pyramid layer
        x1 = self.up_relu(self.up_bn(self.up(x1)))

        # cross connection from encoder
        x2 = self.cross_relu(self.cross_bn(self.conv(x2)))

        # addition
        x = x1 + x2

        # return the computation of the layer
        return self.out_relu(self.out_bn(self.out_conv(x)))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, level):
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


class CrossUpSamp(nn.Module):
    def __init__(self, in_channels, out_channels, mesh_lvl, up=MeshConvTranspose):
        super(CrossUpSamp, self).__init__()

        self.block = nn.Sequential(
            up(in_channels, out_channels, mesh_lvl, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SphericalFPNet(nn.Module):
    def __init__(self, in_ch, out_ch, up="zero-pad", max_level=5, min_level=0, fdim=32, fpn_dim=256, sdim=128):
        # make a call to the parent class constructor
        super(SphericalFPNet, self).__init__()

        # initialise the instance variables
        self.sdim = sdim
        self.fdim = fdim
        self.upsample = MeshConvTranspose if up == "zero-pad" else MeshConvTransposeBilinear
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level

        # initial conv
        self.in_conv = MeshConv(in_ch, fdim, max_level, stride=1)
        self.in_bn = nn.BatchNorm1d(fdim)
        self.in_relu = nn.ReLU(inplace=True)

        # initialise lists to store the encoder, pyramid, and decoder stages
        self.down, self.up, self.cross = [], [], []

        # encoder
        for i in range(self.levels):
            # compute the number of in, out channels, and level
            ch_in = int(fdim * (2 ** i))
            ch_out = int(fdim * (2 ** (i + 1)))
            lvl = max_level - i - 1

            # add a downsample block (512 at L0)
            if i == (self.levels - 1) and min_level == 0:
                self.down.append(Down(ch_in, ch_in, lvl))
                print(f"encoder: {ch_in}-{ch_in}-{lvl}")
            else:
                self.down.append(Down(ch_in, ch_out, lvl))
                print(f"encoder: {ch_in}-{ch_out}-{lvl}")

        # number of channels at lowest level
        in_ch = ch_out if min_level != 0 else ch_in

        # 1x1 cross connection at lowest level to start fpn
        self.cross_conv = nn.Conv1d(in_ch, fpn_dim, kernel_size=1, stride=1)
        self.cross_bn = nn.BatchNorm1d(fpn_dim)
        self.cross_relu = nn.ReLU(inplace=True)

        # feature pyramid
        for i in range(self.levels):
            # compute the number of in, out channels, and level
            ch_in = int(fdim * (2 ** (self.levels - i - 1)))
            ch_out = fpn_dim
            lvl = min_level + i + 1

            # add an upsample block
            self.up.append(Up(ch_in, ch_out, lvl, up=self.upsample))

        # decoder
        for i in range(min_level, max_level + 1):
            # compute the difference in levels
            lvl_diff = max_level - i

            # list to store upsampling stages
            modules = []

            # check if the difference is non zero
            if lvl_diff > 0:
                # add the required number of upsampling stages
                for j in range(i, max_level):
                    if i == j:
                        modules.append(CrossUpSamp(fpn_dim, self.sdim, j + 1, up=self.upsample))
                    else:
                        modules.append(CrossUpSamp(self.sdim, self.sdim, j + 1, up=self.upsample))
            else:
                modules = [nn.Conv1d(fpn_dim, self.sdim, kernel_size=1, stride=1)]

            # add the moddules to the global list for the decoding stage
            self.cross.append(nn.Sequential(*modules))

        # final conv
        self.out_conv = MeshConv(self.sdim, out_ch, max_level)
        self.out_bn = nn.BatchNorm1d(out_ch)

        # initialise the modules
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.cross = nn.ModuleList(self.cross)

    def forward(self, x):
        # in conv
        x_d = [self.in_relu(self.in_bn(self.in_conv(x)))]

        # encoder
        for i in range(self.levels):
            x_d.append(self.down[i](x_d[-1]))

        # initial cross connection at lowest level to start fpn
        x_u = [self.cross_relu(self.cross_bn(self.cross_conv(x_d[-1])))]

        # feature pyramid
        for i in range(self.levels):
            x_u.append(self.up[i](x_u[-1], x_d[self.levels - (i + 1)]))

        # initialise a list to store the final feature maps (all at max level)
        x_c = []

        # decoder
        for i in range(self.levels + 1):
            # grab the appropriate pyramid feature map
            x = x_u[i]

            # loop through the upsampling stages
            for module in self.cross[i]:
                x = module(x)

            # add the current decoder feature map to the global list
            x_c.append(x)

        # convert from list to tensor
        x_c = torch.stack(x_c, dim=0)

        # add the decoder feature maps
        x = torch.sum(x_c, dim=0)

        # out conv
        x = self.out_bn(self.out_conv(x))

        # return the output of the model
        return x


if __name__ == "__main__":
    # from torch.profiler import profile, ProfilerActivity
    from torchinfo import summary
    import torch

    model = SphericalFPNet(4, 15, max_level=5, min_level=0, fdim=32, up="bilinear")

    inputs = torch.randn(1, 4, 10242)

    summary(model, input_size=(1, 4, 10242))
