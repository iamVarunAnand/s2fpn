# import the necessary packages
from ..layers import MeshConv, ResBlock, MeshConvTranspose, UpSamp
from torch import nn


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, bias=True):
        """
            use mesh_file for the mesh of one-level up
        """

        # make a call to the parent constructor
        super(Up, self).__init__()

        # MESHCONV.T
        self.up = MeshConvTranspose(out_ch, out_ch, level, stride=2)
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


class CrossUpSamp(nn.Module):
    def __init__(self, in_channels, out_channels, mesh_lvl):
        super(CrossUpSamp, self).__init__()

        self.block = nn.Sequential(
            MeshConvTranspose(in_channels, out_channels, mesh_lvl, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SphericalFPNet(nn.Module):
    def __init__(self, in_ch, out_ch, max_level=5, min_level=0, fdim=16, fpn_dim=256, sdim=128):
        # make a call to the parent class constructor
        super(SphericalFPNet, self).__init__()

        # initialise the instance variables
        self.sdim = sdim
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level

        # initialise lists to store the encoder and decoder stages
        self.down, self.up = [], []

        # initial conv
        self.in_conv = MeshConv(in_ch, fdim, max_level, stride=1)
        self.in_bn = nn.BatchNorm1d(fdim)
        self.in_relu = nn.ReLU(inplace=True)

        # final conv + upsample
        self.out_conv = nn.Conv1d(self.sdim, out_ch, kernel_size=1, stride=1)
        self.out_bn_a = nn.BatchNorm1d(out_ch)
        self.out_relu_a = nn.ReLU(inplace=True)
        self.out_up_a = Up(64, out_ch, max_level - 1)
        self.out_bn_b = nn.BatchNorm1d(out_ch)
        self.out_relu_b = nn.ReLU(inplace=True)
        self.out_up_b = Up(32, out_ch, max_level)

        # self.out_conv_a = MeshConvTranspose(self.sdim, out_ch, max_level - 1, stride=2)
        # self.out_conv_b = MeshConvTranspose(out_ch, out_ch, max_level, stride=2)

        # backbone
        for i in range(self.levels):
            # compute the number of in, out channels, and level
            ch_in = int(fdim * (2 ** i))
            ch_out = int(fdim * (2 ** (i + 1)))
            lvl = max_level - i - 1

            # add a downsample block (512 at L0)
            if i == (self.levels - 1):
                self.down.append(Down(ch_in, ch_in, lvl))
            else:
                self.down.append(Down(ch_in, ch_out, lvl))

        # 1x1 cross connection at lvl-0
        self.cross_conv = nn.Conv1d(ch_in, fpn_dim, kernel_size=1, stride=1)

        # feature pyramid
        for i in range(3):
            # compute the number of in, out channels, and level
            ch_in = int(fdim * (2 ** (self.levels - i - 1)))
            ch_out = fpn_dim
            lvl = min_level + i + 1

            # add an upsample block
            self.up.append(Up(ch_in, ch_out, lvl))

        # upsampling convolutions for detection stage
        self.conv_1a = CrossUpSamp(fpn_dim, self.sdim, 1)
        self.conv_1b = CrossUpSamp(self.sdim, self.sdim, 2)
        self.conv_1c = CrossUpSamp(self.sdim, self.sdim, 3)
        self.conv_2a = CrossUpSamp(fpn_dim, self.sdim, 2)
        self.conv_2b = CrossUpSamp(self.sdim, self.sdim, 3)
        self.conv_3a = CrossUpSamp(fpn_dim, self.sdim, 3)
        self.conv_4a = nn.Conv1d(fpn_dim, self.sdim, kernel_size=1, stride=1)

        # initialise the modules
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        # pass through initial MESHCONV
        x_d = [self.in_relu(self.in_bn(self.in_conv(x)))]

        # loop through and pass the input through the encoder
        for i in range(self.levels):
            x_d.append(self.down[i](x_d[-1]))

        # initial cross connection at lvl-0
        x_u = [self.cross_conv(x_d[-1])]

        # feature pyramid
        x_u.append(self.up[0](x_u[-1], x_d[self.levels - 1]))
        x_u.append(self.up[1](x_u[-1], x_d[self.levels - 2]))
        x_u.append(self.up[2](x_u[-1], x_d[self.levels - 3]))

        # detection stage
        x1 = self.conv_1c(self.conv_1b(self.conv_1a(x_u[0])))
        x2 = self.conv_2b(self.conv_2a(x_u[1]))
        x3 = self.conv_3a(x_u[2])
        x4 = self.conv_4a(x_u[3])

        # add all the pyramid levels together
        x = x1 + x2 + x3 + x4

        # conv + 4x upsample for final prediction
        # x = self.out_conv_b(self.out_conv_a(x))
        x = self.out_relu_a(self.out_bn_a(self.out_conv(x)))
        x = self.out_relu_b(self.out_bn_b(self.out_up_a(x, x_d[1])))
        x = self.out_up_b(x, x_d[0])

        # return the output of the model
        return x


class SphericalFPNetLarge(nn.Module):
    def __init__(self, in_ch, out_ch, max_level=5, min_level=0, fdim=16, fpn_dim=256, sdim=128):
        # make a call to the parent class constructor
        super(SphericalFPNetLarge, self).__init__()

        # initialise the instance variables
        self.sdim = sdim
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level

        # initialise lists to store the encoder and decoder stages
        self.down, self.up = [], []

        # initial conv
        self.in_conv = MeshConv(in_ch, fdim, max_level, stride=1)
        self.in_bn = nn.BatchNorm1d(fdim)
        self.in_relu = nn.ReLU(inplace=True)

        # final conv
        self.out_conv = MeshConv(self.sdim, out_ch, max_level)
        self.out_bn = nn.BatchNorm1d(out_ch)
        # self.out_conv = nn.Conv1d(self.sdim, out_ch, kernel_size=1, stride=1)
        # self.out_bn_a = nn.BatchNorm1d(out_ch)
        # self.out_relu_a = nn.ReLU(inplace=True)
        # self.out_up_a = Up(64, out_ch, max_level - 1)
        # self.out_bn_b = nn.BatchNorm1d(out_ch)
        # self.out_relu_b = nn.ReLU(inplace=True)
        # self.out_up_b = Up(32, out_ch, max_level)

        # self.out_conv_a = MeshConvTranspose(self.sdim, out_ch, max_level - 1, stride=2)
        # self.out_conv_b = MeshConvTranspose(out_ch, out_ch, max_level, stride=2)

        # backbone
        for i in range(self.levels):
            # compute the number of in, out channels, and level
            ch_in = int(fdim * (2 ** i))
            ch_out = int(fdim * (2 ** (i + 1)))
            lvl = max_level - i - 1

            # add a downsample block (512 at L0)
            if i == (self.levels - 1):
                self.down.append(Down(ch_in, ch_in, lvl))
            else:
                self.down.append(Down(ch_in, ch_out, lvl))

        # 1x1 cross connection at lvl-0
        self.cross_conv = nn.Conv1d(ch_in, fpn_dim, kernel_size=1, stride=1)
        self.cross_bn = nn.BatchNorm1d(fpn_dim)
        self.cross_relu = nn.ReLU(inplace=True)

        # feature pyramid
        for i in range(3):
            # compute the number of in, out channels, and level
            ch_in = int(fdim * (2 ** (self.levels - i - 1)))
            ch_out = fpn_dim
            lvl = min_level + i + 1

            # add an upsample block
            self.up.append(Up(ch_in, ch_out, lvl))

        # upsampling convolutions for detection stage
        self.conv_1a = CrossUpSamp(fpn_dim, self.sdim, min_level + 1)
        self.conv_1b = CrossUpSamp(self.sdim, self.sdim, min_level + 2)
        self.conv_1c = CrossUpSamp(self.sdim, self.sdim, min_level + 3)
        self.conv_2a = CrossUpSamp(fpn_dim, self.sdim, min_level + 2)
        self.conv_2b = CrossUpSamp(self.sdim, self.sdim, min_level + 3)
        self.conv_3a = CrossUpSamp(fpn_dim, self.sdim, min_level + 3)
        self.conv_4a = nn.Conv1d(fpn_dim, self.sdim, kernel_size=1, stride=1)

        # initialise the modules
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        # pass through initial MESHCONV
        x_d = [self.in_relu(self.in_bn(self.in_conv(x)))]

        # loop through and pass the input through the encoder
        for i in range(self.levels):
            x_d.append(self.down[i](x_d[-1]))

        # initial cross connection at lvl-0
        x_u = [self.cross_relu(self.cross_bn(self.cross_conv(x_d[-1])))]

        # feature pyramid
        x_u.append(self.up[0](x_u[-1], x_d[self.levels - 1]))
        x_u.append(self.up[1](x_u[-1], x_d[self.levels - 2]))
        x_u.append(self.up[2](x_u[-1], x_d[self.levels - 3]))

        # detection stage
        x1 = self.conv_1c(self.conv_1b(self.conv_1a(x_u[0])))
        x2 = self.conv_2b(self.conv_2a(x_u[1]))
        x3 = self.conv_3a(x_u[2])
        x4 = self.conv_4a(x_u[3])

        # add all the pyramid levels together
        x = x1 + x2 + x3 + x4

        # conv + 4x upsample for final prediction
        x = self.out_bn(self.out_conv(x))
        # # x = self.out_conv_b(self.out_conv_a(x))
        # x = self.out_relu_a(self.out_bn_a(self.out_conv(x)))
        # x = self.out_relu_b(self.out_bn_b(self.out_up_a(x, x_d[1])))
        # x = self.out_up_b(x, x_d[0])

        # return the output of the model
        return x


if __name__ == "__main__":
    # from torch.profiler import profile, ProfilerActivity
    from torchinfo import summary
    import torch

    model = SphericalFPNetLarge(4, 15, min_level=2, fdim=32)
    inputs = torch.randn(1, 4, 10242)

    summary(model, input_size=(1, 4, 10242))

#     # writer = SummaryWriter('logs')
#     # writer.add_graph(model, inputs)

#     # # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
#     # #     model(inputs)

#     # # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
