# import the necessary packages
from matplotlib.cbook import normalize_kwargs
from uscnn.layers import MeshConvTest, MeshConv
from uscnn.utils import Icosphere, spmatmul, sparse2tensor
from tqdm import tqdm
import meshplot as mp
import numpy as np
import torch
import glob
import igl
import os

mp.offline()
np.random.seed(7)

# grab the path to the dataset files
paths = list(glob.glob("/home/varun/datasets/2d3ds_sphere/area_1/*.npz"))

# sort the paths according to the split
paths = sorted(paths, key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0].split("_")[-1]))


def normalize(vectors, radius=1):
    '''
    Reproject to spherical surface
    '''

    scalar = (vectors ** 2).sum(axis=1)**.5
    unit = vectors / scalar.reshape((-1, 1))
    offset = radius - scalar

    return vectors + unit * offset.reshape((-1, 1))


def normalize_features(x):
    return (x - x.min()) / (x.max() - x.min())


def upsamp(img, level):
    nv_pad = 30 * (4**level)
    n_channels = 3

    # initialise an icosphere
    ico = Icosphere(level=level)
    ico_up = Icosphere(level=level + 1)

    img = img[:, :n_channels]

    img = img[:ico.vertices.shape[0]]

    zeros_pad = np.zeros((nv_pad, n_channels))
    img = np.concatenate((img, zeros_pad))

    # # bilinear interpolation upsampling
    # img[ico_up.faces[3:][::4, 0]] = (img[ico.faces[:, 0]] + img[ico.faces[:, 1]]) / 2
    # img[ico_up.faces[3:][::4, 1]] = (img[ico.faces[:, 1]] + img[ico.faces[:, 2]]) / 2
    # img[ico_up.faces[3:][::4, 2]] = (img[ico.faces[:, 2]] + img[ico.faces[:, 1]]) / 2

    # reshape
    img_ = img.reshape((1, 3, ico_up.nv))
    img_ = torch.FloatTensor(img_)

    out = np.squeeze(MeshConvTest(3, 1, mesh_lvl=(level + 1))(img_).detach().numpy())
    out = out.reshape((4, ico_up.nv, 3))

    G = igl.grad(ico_up.vertices, ico_up.faces)
    grad_face = G.dot(img).reshape((*ico_up.faces.shape, 3), order="F")

    # face gradients along cardinal directions
    grad_face_ew = np.sum(np.multiply(grad_face, ico_up.EW[..., None]), keepdims=False, axis=-1).T
    grad_face_ns = np.sum(np.multiply(grad_face, ico_up.NS[..., None]), keepdims=False, axis=-1).T

    grad_face_ew = torch.FloatTensor(grad_face_ew[None, ...])
    grad_face_ns = torch.FloatTensor(grad_face_ns[None, ...])

    F2V = sparse2tensor(ico_up.F2V.tocoo())

    # vertex gradients (weighted by face area)
    grad_vert_ew = spmatmul(grad_face_ew, F2V).numpy().squeeze().T
    grad_vert_ns = spmatmul(grad_face_ns, F2V).numpy().squeeze().T

    # # plot the image and save it to disk
    # output_img_path = f"figs/op/bl_identity.html"
    # mp.plot(ico_up.vertices, ico_up.faces, c=out[0, :ico_up.vertices.shape[0]],
    #         shading={"point_size": 0.5}, filename=output_img_path)

    # output_img_path = f"figs/op/bl_laplacian.html"
    # mp.plot(ico_up.vertices, ico_up.faces, c=np.mean(out[1, :ico_up.vertices.shape[0]], axis=-1),
    #         shading={"point_size": 0.5, "colormap": "gray"}, filename=output_img_path)

    # output_img_path = f"figs/op/bl_ew.html"
    # mp.plot(ico_up.vertices, ico_up.faces, c=np.mean(grad_vert_ew, axis=-1), shading={
    #         "point_size": 0.5, "colormap": "gray"}, filename=output_img_path)

    # output_img_path = f"figs/op/bl_ns.html"
    # mp.plot(ico_up.vertices, ico_up.faces, c=np.mean(grad_vert_ns, axis=-1), shading={
    #         "point_size": 0.5, "colormap": "gray"}, filename=output_img_path)

    output_img_path = f"figs/op/bl_out_{level + 1}.html"

    identity = normalize_features(out[0])
    laplacian = normalize_features(out[1])
    grad_vert_ew = normalize_features(grad_vert_ew)
    grad_vert_ns = normalize_features(grad_vert_ns)

    t = identity + laplacian + grad_vert_ew + grad_vert_ns
    t = normalize_features(t)

    mp.plot(ico_up.vertices, ico_up.faces, c=t, shading={"point_size": 0.5}, filename=output_img_path)

    # return out[0] + out[1] + grad_vert_ew + grad_vert_ns
    return t


def downsamp(img, level):
    nv_pad = 30 * (4**level)
    n_channels = 3

    # initialise an icosphere
    ico = Icosphere(level=level)
    ico_down = Icosphere(level=level - 1)

    img = img[:, :n_channels]

    # downsample
    img = img[:ico_down.vertices.shape[0]]

    # reshape
    img_ = img.reshape((1, 3, ico_down.nv))
    img_ = torch.FloatTensor(img_)

    out = np.squeeze(MeshConvTest(3, 1, mesh_lvl=(level - 1))(img_).detach().numpy())
    out = out.reshape((4, ico_down.nv, 3))

    G = igl.grad(ico_down.vertices, ico_down.faces)
    grad_face = G.dot(img).reshape((*ico_down.faces.shape, 3), order="F")

    # face gradients along cardinal directions
    grad_face_ew = np.sum(np.multiply(grad_face, ico_down.EW[..., None]), keepdims=False, axis=-1).T
    grad_face_ns = np.sum(np.multiply(grad_face, ico_down.NS[..., None]), keepdims=False, axis=-1).T

    grad_face_ew = torch.FloatTensor(grad_face_ew[None, ...])
    grad_face_ns = torch.FloatTensor(grad_face_ns[None, ...])

    F2V = sparse2tensor(ico_down.F2V.tocoo())

    # vertex gradients (weighted by face area)
    grad_vert_ew = spmatmul(grad_face_ew, F2V).numpy().squeeze().T
    grad_vert_ns = spmatmul(grad_face_ns, F2V).numpy().squeeze().T

    # plot the image and save it to disk
    # output_img_path = f"figs/op/bl_identity.html"
    # mp.plot(ico_down.vertices, ico_down.faces, c=out[0, :ico_down.vertices.shape[0]],
    #         shading={"point_size": 0.5}, filename=output_img_path)

    # output_img_path = f"figs/op/bl_laplacian.html"
    # mp.plot(ico_down.vertices, ico_down.faces, c=np.mean(out[1, :ico_down.vertices.shape[0]], axis=-1),
    #         shading={"point_size": 0.5, "colormap": "gray"}, filename=output_img_path)

    # output_img_path = f"figs/op/bl_ew.html"
    # mp.plot(ico_down.vertices, ico_down.faces, c=np.mean(grad_vert_ew, axis=-1), shading={
    #         "point_size": 0.5, "colormap": "gray"}, filename=output_img_path)

    # output_img_path = f"figs/op/bl_ns.html"
    # mp.plot(ico_down.vertices, ico_down.faces, c=np.mean(grad_vert_ns, axis=-1), shading={
    #         "point_size": 0.5, "colormap": "gray"}, filename=output_img_path)

    output_img_path = f"figs/op/down_out_{level - 1}.html"

    identity = normalize_features(out[0])
    laplacian = normalize_features(out[1])
    grad_vert_ew = normalize_features(grad_vert_ew)
    grad_vert_ns = normalize_features(grad_vert_ns)

    t = identity + laplacian + grad_vert_ew + grad_vert_ns
    t = normalize_features(t)

    mp.plot(ico_down.vertices, ico_down.faces, c=t, shading={"point_size": 0.5}, filename=output_img_path)

    # return out[0] + out[1] + grad_vert_ew + grad_vert_ns
    return t


if __name__ == "__main__":
    # load a sample image
    arr = np.load(paths[0])
    img, lbl = arr["data"], arr["labels"]

    level = 4
    img_up = img
    for lvl in range(3, level):
        img_up = upsamp(img_up, lvl)

    level += 1
    for lvl in reversed(range(level, 8)):
        img_down = downsamp(img, lvl)

    out = img_up + img_down
    out = normalize_features(out)

    output_img_path = f"figs/op/add_out_{level - 1}.html"
    ico_down = Icosphere(level=level - 1)
    mp.plot(ico_down.vertices, ico_down.faces, c=np.mean(out, axis=-1),
            shading={"point_size": 0.5, "colormap": "gray"}, filename=output_img_path)
