# import the necessary packages
from ..layers import MeshConv, MeshConvTranspose, ResBlock, UpSamp
from torch import nn


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, bias=True):
        """
            use mesh_file for the mesh of one-level up
        """

        # make a call to the parent constructor
        super(Up, self).__init__()

        # MESHCONV.T
        self.up = UpSamp(level)

        # cross connection
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1)

        # final meshconv
        self.out_conv = MeshConv(out_ch, out_ch, level)

    def forward(self, x1, x2):
        # upsample the previous pyramid layer
        x1 = self.up(x1)

        # cross connection from encoder
        x2 = self.conv(x2)

        # addition
        x = x1 + x2

        # return the computation of the layer
        return self.out_conv(x)


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
            MeshConv(in_channels, out_channels, mesh_lvl - 1, stride=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            UpSamp(mesh_lvl)
        )

    def forward(self, x):
        return self.block(x)


class SphericalFPNet(nn.Module):
    def __init__(self, in_ch, out_ch, max_level=5, min_level=0, fdim=16, fpn_dim=256):
        # make a call to the parent class constructor
        super(SphericalFPNet, self).__init__()

        # initialise the instance variables
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level

        # initialise lists to store the encoder and decoder stages
        self.down, self.up = [], []

        # initial conv
        self.in_conv = MeshConv(in_ch, fdim, max_level, stride=1)

        # final conv + upsample
        self.out_conv = nn.Conv1d(128, out_ch, kernel_size=1, stride=1)
        self.out_up_a = UpSamp(max_level - 1)
        self.out_up_b = UpSamp(max_level)

        # backbone
        for i in range(self.levels):
            # compute the number of in, out channels, and level
            ch_in = int(fdim * (2 ** i))
            ch_out = int(fdim * (2 ** (i + 1)))
            lvl = max_level - i - 1

            # add a downsample block
            self.down.append(Down(ch_in, ch_out, lvl))

        # 1x1 cross connection at lvl-0
        self.cross_conv = nn.Conv1d(ch_out, fpn_dim, kernel_size=1, stride=1)

        # feature pyramid
        for i in range(3):
            # compute the number of in, out channels, and level
            ch_in = int(fdim * (2 ** (self.levels - i - 1)))
            ch_out = fpn_dim
            lvl = min_level + i + 1

            # add an upsample block
            self.up.append(Up(ch_in, ch_out, lvl))

        # upsampling convolutions for detection stage
        self.conv_1a = CrossUpSamp(fpn_dim, 128, 1)
        self.conv_1b = CrossUpSamp(128, 128, 2)
        self.conv_1c = CrossUpSamp(128, 128, 3)
        self.conv_2a = CrossUpSamp(fpn_dim, 128, 2)
        self.conv_2b = CrossUpSamp(128, 128, 3)
        self.conv_3a = CrossUpSamp(fpn_dim, 128, 3)
        self.conv_4a = nn.Conv1d(fpn_dim, 128, kernel_size=1, stride=1)

        # initialise the modules
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        # pass through initial MESHCONV
        x_d = [self.in_conv(x)]

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
        x = self.out_conv(x)
        x = self.out_up_b(self.out_up_a(x))

        # return the output of the model
        return x


if __name__ == "__main__":
    # from torch.profiler import profile, ProfilerActivity
    from torchinfo import summary
    import torch

    model = SphericalFPNet(4, 15, fdim=32).to(torch.device("cpu"))
    inputs = torch.randn(1, 4, 10242).to(torch.device("cpu"))

    summary(model, input_size=(1, 4, 10242))

    # writer = SummaryWriter('logs')
    # writer.add_graph(model, inputs)

    # # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    # #     model(inputs)

    # # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
