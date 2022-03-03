# import the necessary packages
from uscnn.utils import Icosphere
from tqdm import tqdm
import meshplot as mp
import numpy as np
import glob
import os

mp.offline()

np.random.seed(7)


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
    paths = list(glob.glob(
        r"C:\Users\spike\Documents\Python Scripts\MLP_CW4\ugscnn-master\ugscnn-master\new\ugscnn\experiments\exp3_2d3ds\2d3ds_sphere\area_1\*.npz"))

    # sort the paths according to the split
    paths = sorted(paths, key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0].split("_")[-1]))

    # constants
    level = 6
    nv_pad = 30 * (4**level)

    # initialise an icosphere
    ico = Icosphere(level=level)
    ico_up = Icosphere(level=level + 1)

    # load a sample image
    arr = np.load(paths[0])
    img, lbl = arr["data"], arr["labels"]


    img = img[:ico.vertices.shape[0]]
    zeros_pad = np.zeros((nv_pad, 4))
    img = np.concatenate((img, zeros_pad))

    #bilinear interpolation upsampling
    img[ico_up.faces[3:][::4, 0]] = (img[ico.faces[:, 0]] + img[ico.faces[:, 1]]) / 2
    img[ico_up.faces[3:][::4, 1]] = (img[ico.faces[:, 1]] + img[ico.faces[:, 2]]) / 2
    img[ico_up.faces[3:][::4, 2]] = (img[ico.faces[:, 2]] + img[ico.faces[:, 1]]) / 2

    # plot the image and save it to disk
    output_img_path = f"figs/area_1/{paths[0].split(os.path.sep)[-1].split('.')[0]}_upsample_test.html"
    mp.plot(ico_up.vertices, ico_up.faces, c=img[:ico_up.vertices.shape[0], :3], shading={"point_size": 0.5}, filename=output_img_path)

    output_img_path = f"figs/area_1/{paths[0].split(os.path.sep)[-1].split('.')[0]}_test.html"
    mp.plot(ico.vertices, ico.faces, c=img[:ico.vertices.shape[0], :3], filename=output_img_path)

