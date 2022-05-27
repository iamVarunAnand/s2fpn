# import the necessary packages
from scipy import sparse
from sklearn import neighbors
from uscnn.utils import Icosphere, sparse2tensor, spmatmul
from tqdm import tqdm
import matplotlib.pyplot as plt
import meshplot as mp
import numpy as np
import torch
import glob
import os

mp.offline()

# np.random.seed(7)

np.random.seed(42)
COLORS = {
    k: np.random.uniform(0, 1, size=(3,)) for k in range(15)
}


def normalize(vectors, radius=1):
    '''
    Reproject to spherical surface
    '''

    scalar = (vectors ** 2).sum(axis=1)**.5
    unit = vectors / scalar.reshape((-1, 1))
    offset = radius - scalar

    return vectors + unit * offset.reshape((-1, 1))


if __name__ == "__main__":
    # grab the path to the dataset files
    paths = list(glob.glob("/home/varun/datasets/2d3ds_sphere/area_1/*.npz"))

    # sort the paths according to the split
    paths = sorted(paths, key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0].split("_")[-1]))

    # constants
    level = 7
    nv_pad = 30 * (4**level)

    # initialise an icosphere
    ico = Icosphere(level=level)
    ico_down = Icosphere(level=level - 1)

    # # print(ico_down.vertices.shape)

    # # load a sample image
    # arr = np.load(paths[0])
    # img, lbl = arr["data"][:ico.nv, :3], arr["labels"]

    # out = np.zeros((ico_down.nv, 3))
    # # # out[ico.faces[3:][::4, 0], 2] = 255

    # stack1 = np.stack([ico_down.faces[:, 0], ico.faces[3:][::4, 0], ico.faces[3:][::4, 2]], axis=-1)
    # stack2 = np.stack([ico_down.faces[:, 1], ico.faces[3:][::4, 0], ico.faces[3:][::4, 1]], axis=-1)
    # stack3 = np.stack([ico_down.faces[:, 2], ico.faces[3:][::4, 1], ico.faces[3:][::4, 2]], axis=-1)
    # stack = np.vstack([stack1, stack2, stack3])

    # stack = stack[stack[:, 0].argsort()]

    # u, idx = np.unique(stack[:, 0], return_index=True)
    # splits = np.split(stack[:, 1:], idx[1:])

    # sd, si, sj = [], [], []
    # for i, s in enumerate(splits):
    #     neighbours = np.unique(s.flatten())

    #     for vertex in neighbours:
    #         sd.append(1 / neighbours.shape[0])
    #         si.append(i)
    #         sj.append(vertex)

    # print(len(sd), len(si), len(sj))

    # VA = sparse.coo_matrix((sd, (si, sj)), shape=(ico_down.nv, ico.nv))
    # VA = sparse2tensor(VA)
    # out = spmatmul(torch.Tensor(img.T[None, ...]), VA).numpy()[0].T

    # print(out.shape)

    # # out[i] = np.mean(img[v], axis=0)

    # # plot the image and save it to disk
    # output_img_path = f"figs/avg_downsample.html"
    # mp.plot(ico_down.vertices, ico_down.faces, c=out, filename=output_img_path)

    # # output_img_path = f"figs/original.html"
    # # mp.plot(ico.vertices, ico.faces, c=img, filename=output_img_path)

    # # output_img_path = f"figs/drop_downsample.html"
    # # mp.plot(ico_down.vertices, ico_down.faces, c=img[:ico_down.nv], filename=output_img_path)
