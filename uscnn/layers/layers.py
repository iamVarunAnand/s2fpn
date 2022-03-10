# import the necessary packages
from ..utils import sparse2tensor, spmatmul
from torch.nn.parameter import Parameter
from ..meshes import MESHES
from torch import nn
import torch
import math


class _MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, mesh_lvl, stride=1, bias=True):
        # assert for supported strides
        assert stride in [1, 2]

        # make a call to the parent constructor
        super(_MeshConv, self).__init__()

        # initialise the instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncoeff = 4

        # check to see if bias needs to be used
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # initialise the PDO parameters (trainable)
        self.coeffs = Parameter(torch.Tensor(out_channels, in_channels, self.ncoeff))
        self.initialise_weights()

        # grab the mesh file
        pkl = MESHES[mesh_lvl]
        self.pkl = pkl
        self.nv = pkl["nv"]

        # extract the required matrices
        G = pkl["G"]  # gradient matrix V->F, 3#F x #V
        NS = pkl["NS"]  # north-south vector field, #F x 3
        EW = pkl["EW"]  # east-west vector field, #F x 3

        # register matrices as non-trainable parameters
        self.register_buffer("G", G)
        self.register_buffer("NS", NS)
        self.register_buffer("EW", EW)

    def initialise_weights(self):
        # compute the standard deviation of weight distribution
        n = self.in_channels * self.ncoeff
        stdv = 1. / math.sqrt(n)

        # glorot uniform initialisation
        self.coeffs.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class MeshConv(_MeshConv):
    def __init__(self, in_channels, out_channels, mesh_lvl, stride=1, bias=True):
        # make a call to the parent class constructor
        super(MeshConv, self).__init__(in_channels, out_channels, mesh_lvl, stride, bias)
        pkl = self.pkl

        if stride == 2:
            self.nv_prev = pkl["nv_prev"]
            L = sparse2tensor(pkl["L"].tocsr()[:self.nv_prev].tocoo())  # laplacian matrix V->V
            F2V = sparse2tensor(pkl["F2V"].tocsr()[:self.nv_prev].tocoo())  # F->V, #V x #F
        else:  # stride == 1
            self.nv_prev = pkl["nv"]
            L = sparse2tensor(pkl["L"].tocoo())
            F2V = sparse2tensor(pkl["F2V"].tocoo())

        self.register_buffer("L", L)
        self.register_buffer("F2V", F2V)

    def forward(self, input):
        # gradient
        grad_face = spmatmul(input, self.G)
        grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2)  # gradient, 3 component per face

        # laplacian
        laplacian = spmatmul(input, self.L)

        # identity
        identity = input[..., :self.nv_prev]

        # face gradients along cardinal directions
        grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
        grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)

        # vertex gradients (weighted by face area)
        grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
        grad_vert_ns = spmatmul(grad_face_ns, self.F2V)

        # features
        feat = [identity, laplacian, grad_vert_ew, grad_vert_ns]

        # dot product to compute the PDO convolution
        out = torch.stack(feat, dim=-1)
        out = torch.sum(torch.sum(torch.mul(out.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
        out += self.bias.unsqueeze(-1)

        # return the computed feature maps
        return out


class MeshConvTest(_MeshConv):
    def __init__(self, in_channels, out_channels, mesh_lvl, stride=1, bias=True):
        # make a call to the parent class constructor
        super(MeshConvTest, self).__init__(in_channels, out_channels, mesh_lvl, stride, bias)
        pkl = self.pkl

        if stride == 2:
            self.nv_prev = pkl["nv_prev"]
            L = sparse2tensor(pkl["L"].tocsr()[:self.nv_prev].tocoo())  # laplacian matrix V->V
            F2V = sparse2tensor(pkl["F2V"].tocsr()[:self.nv_prev].tocoo())  # F->V, #V x #F
        else:  # stride == 1
            self.nv_prev = pkl["nv"]
            L = sparse2tensor(pkl["L"].tocoo())
            F2V = sparse2tensor(pkl["F2V"].tocoo())

        self.register_buffer("L", L)
        self.register_buffer("F2V", F2V)

    def forward(self, input):
        # identity
        identity = input[..., :self.nv_prev]

        # gradient
        grad_face = spmatmul(input, self.G)
        grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2)  # gradient, 3 component per face

        # laplacian
        laplacian = spmatmul(input, self.L)

        # face gradients along cardinal directions
        grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
        grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)

        # vertex gradients (weighted by face area)
        grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
        grad_vert_ns = spmatmul(grad_face_ns, self.F2V)

        # features
        feat = [identity, laplacian, grad_vert_ew, grad_vert_ns]

        # # dot product to compute the PDO convolution
        # out = torch.stack(feat, dim=-1)
        # out = torch.sum(torch.sum(torch.mul(out.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
        # out += self.bias.unsqueeze(-1)

        # return the computed feature maps
        return torch.stack(feat, dim=0)


class MeshConvTranspose(_MeshConv):
    def __init__(self, in_channels, out_channels, mesh_lvl, stride=2, bias=True):
        # assert for supported strides
        assert(stride == 2)

        # make a call to the parent class constructor
        super(MeshConvTranspose, self).__init__(in_channels, out_channels, mesh_lvl, stride, bias)

        pkl = self.pkl
        self.nv_prev = self.pkl["nv_prev"]
        self.nv_pad = self.nv - self.nv_prev

        L = sparse2tensor(pkl["L"].tocoo())  # laplacian matrix V->V
        F2V = sparse2tensor(pkl["F2V"].tocoo())  # F->V, #V x #F

        self.register_buffer("L", L)
        self.register_buffer("F2V", F2V)

    def forward(self, input):
        # pad input with zeros to match the next mesh resolution
        ones_pad = torch.ones(*input.size()[:2], self.nv_pad).to(input.device)
        input = torch.cat((input, ones_pad), dim=-1)

        # gradient
        grad_face = spmatmul(input, self.G)
        grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2)  # gradient, 3 component per face

        # laplacian
        laplacian = spmatmul(input, self.L)

        # identity
        identity = input

        # face gradients along cardinal directions
        grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
        grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)

        # vertex gradients (weighted by face area)
        grad_vert_ew = spmatmul(grad_face_ew, self.F2V)
        grad_vert_ns = spmatmul(grad_face_ns, self.F2V)

        # features
        feat = [identity, laplacian, grad_vert_ew, grad_vert_ns]

        # dot product to compute the PDO convolution
        out = torch.stack(feat, dim=-1)
        out = torch.sum(torch.sum(torch.mul(out.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
        out += self.bias.unsqueeze(-1)

        # return the computed feature maps
        return out


class DownSamp(nn.Module):
    def __init__(self, nv_prev):
        # make a call to the parent constructor
        super(DownSamp, self).__init__()

        # initialise the instance variables
        self.nv_prev = nv_prev

    def forward(self, x):
        # downsample and return
        return x[..., :self.nv_prev]


class UpSamp(nn.Module):
    def __init__(self, mesh_lvl):
        # make a call to the parent constructor
        super(UpSamp, self).__init__()

        # initialise the instance variables
        ico = MESHES[mesh_lvl - 1]
        ico_up = MESHES[mesh_lvl]
        self.ico = ico
        self.ico_up = ico_up
        self.nv = ico_up["nv"]
        self.nv_prev = ico["nv"]
        self.nv_pad = self.nv - self.nv_prev

    def forward(self, x):
        # upsample and return
        # input has size batch_size x 256 x nv_prev

        ones_pad = torch.ones(*x.size()[:2], self.nv_pad).to(x.device)
        x = torch.cat((x, ones_pad), dim=-1)

        x[:, :, (self.ico_up["F"])[3:][::4, 0]] = (x[:, :, (self.ico["F"])[:, 0]] + x[:, :, (self.ico["F"])[:, 1]]) / 2
        x[:, :, (self.ico_up["F"])[3:][::4, 1]] = (x[:, :, (self.ico["F"])[:, 1]] + x[:, :, (self.ico["F"])[:, 2]]) / 2
        x[:, :, (self.ico_up["F"])[3:][::4, 2]] = (x[:, :, (self.ico["F"])[:, 2]] + x[:, :, (self.ico["F"])[:, 1]]) / 2

        return x


class UpSampNearest(nn.Module):
    def __init__(self, mesh_lvl):
        # make a call to the parent constructor
        super(UpSamp, self).__init__()

        # initialise the instance variables
        ico = MESHES[mesh_lvl - 1]
        ico_up = MESHES[mesh_lvl]
        self.ico = ico
        self.ico_up = ico_up
        self.nv = ico_up["nv"]
        self.nv_prev = ico["nv"]
        self.nv_pad = self.nv - self.nv_prev

    def forward(self, x):
        # upsample and return
        # input has size batch_size x 256 x nv_prev

        ones_pad = torch.ones(*x.size()[:2], self.nv_pad).to(x.device)
        x = torch.cat((x, ones_pad), dim=-1)

        x[:, :, (self.ico_up["F"])[3:][::4, 0]] = x[:, :, (self.ico["F"])[:, 0]]
        x[:, :, (self.ico_up["F"])[3:][::4, 1]] = x[:, :, (self.ico["F"])[:, 1]]
        x[:, :, (self.ico_up["F"])[3:][::4, 2]] = x[:, :, (self.ico["F"])[:, 2]]

        return x


class UpSampPad(nn.Module):
    def __init__(self, mesh_lvl):
        # make a call to the parent constructor
        super(UpSampPad, self).__init__()

        # initialise the instance variables
        ico_up = MESHES[mesh_lvl]
        self.ico_up = ico_up
        self.nv = ico_up["nv"]
        self.nv_prev = ico_up["nv_prev"]
        self.nv_pad = self.nv - self.nv_prev

    def forward(self, x):
        # upsample and return
        # input has size batch_size x 256 x nv_prev

        ones_pad = torch.ones(*x.size()[:2], self.nv_pad).to(x.device)
        x = torch.cat((x, ones_pad), dim=-1)

        return x


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
        # self.bn_1a = nn.BatchNorm1d(neck_chan)
        self.bn_1a = nn.GroupNorm(32, neck_chan)

        # MESHCONV -> BN
        self.conv_2a = MeshConv(neck_chan, neck_chan, mesh_lvl=lvl, stride=1)
        # self.bn_2a = nn.BatchNorm1d(neck_chan)
        self.bn_2a = nn.GroupNorm(32, neck_chan)

        # CONV 1x1 -> BN
        self.conv_3a = nn.Conv1d(neck_chan, out_chan, kernel_size=1, stride=1)
        # self.bn_3a = nn.BatchNorm1d(out_chan)
        self.bn_3a = nn.GroupNorm(32, out_chan)

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
