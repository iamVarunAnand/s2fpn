# import the necessary packages
from ..layers import MeshConv, MeshConvTranspose, ResBlock
from torch import nn
import torch
import os


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
        """
            uses the mesh_file for the mesh of one-level up
        """

        # make a call to the parent constructor
        super(Up, self).__init__()

        # build the path to the mesh file
        mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(level))

        # MESHCONV.T
        half_in = int(in_ch / 2)
        self.up = MeshConvTranspose(half_in, half_in, mesh_file, stride=2)

        # res block
        self.conv = ResBlock(in_ch, out_ch, out_ch, level, False, mesh_folder)

    def forward(self, x1, x2):
        # upsample the previous pyramid level
        x1 = self.up(x1)

        # concatenate with features from encoder stage
        x = torch.cat([x2, x1], dim=1)

        # pass through res block
        x = self.conv(x)

        # return the layer computation
        return x

# class Up(nn.Module):
#     def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
#         """
#         use mesh_file for the mesh of one-level up
#         """
#         super().__init__()
#         mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(level))
#         # half_in = int(in_ch / 2)
#         self.up = MeshConv_transpose(out_ch, out_ch, mesh_file, stride=2)
#         self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x2 = self.conv(x2)
#         x = x1 + x2
#         # x = torch.cat([x2, x1], dim=1)
#         # x = self.conv(x)
#         return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
        """
            use the mesh_file for the mesh of one-level down
        """

        # make a call to the parent constructor
        super(Down, self).__init__()

        # res block
        self.conv = ResBlock(in_ch, in_ch, out_ch, level + 1, True, mesh_folder)

    def forward(self, x):
        # pass the input through the res block and return
        return self.conv(x)


class SphericalUNet(nn.Module):
    def __init__(self, mesh_folder, in_ch, out_ch, max_level=5, min_level=0, fdim=16):
        # make a call to the parent class constructor
        super(SphericalUNet, self).__init__()

        # initialise the instance variables
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level

        # initialise lists to store the encoder and decoder stages
        self.down, self.up = [], []

        # initial and final MESHCONV
        self.in_conv = MeshConv(in_ch, fdim, self.__meshfile(max_level), stride=1)
        self.out_conv = MeshConv(fdim, out_ch, self.__meshfile(max_level), stride=1)

        # encoder
        for i in range(self.levels - 1):
            # compute the number of in, out channels, and level
            ch_in = fdim * (2 ** i)
            ch_out = fdim * (2 ** (i + 1))
            lvl = max_level - i - 1

            # add a downsample block
            self.down.append(Down(ch_in, ch_out, lvl, mesh_folder))

        # bottleneck
        ch_in = fdim * (2 ** (self.levels - 1))
        ch_out = fdim * (2 ** (self.levels - 1))
        self.down.append(Down(ch_in, ch_out, min_level, mesh_folder))

        # decoder
        for i in range(self.levels - 1):
            # compute the number of in, out channels, and level
            ch_in = fdim * (2 ** (self.levels - i))
            ch_out = fdim * (2 ** (self.levels - i - 2))
            lvl = min_level + i + 1

            # add an upsample block
            self.up.append(Up(ch_in, ch_out, lvl, mesh_folder))

        # final upsample
        self.up.append(Up(fdim * 2, fdim, max_level, mesh_folder))

        # initialise the encoder and decoders as nn modules
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def __meshfile(self, i):
        return os.path.join(self.mesh_folder, "icosphere_{}.pkl".format(i))

    def forward(self, x):
        # pass through initial MESHCONV
        x_ = [self.in_conv(x)]

        # loop through and pass the input through the encoder
        for i in range(self.levels):
            x_.append(self.down[i](x_[-1]))

        # first upsample
        x = self.up[0](x_[-1], x_[-2])

        # loop through ans pass through the decoder
        for i in range(self.levels - 1):
            x = self.up[i + 1](x, x_[-3 - i])

        # pass through final MESHCONV
        x = self.out_conv(x)

        # return the output of the model
        return x


# if __name__ == "__main__":
#     from torchinfo import summary

#     model = SphericalUNet("uscnn/meshes", 4, 15, max_level=5, fdim=32)

#     summary(model, input_size=(16, 4, 10242))
