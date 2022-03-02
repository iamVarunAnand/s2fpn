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
    paths = list(glob.glob("/home/varun/datasets/2d3ds_sphere/area_1/*.npz"))

    # sort the paths according to the split
    paths = sorted(paths, key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0].split("_")[-1]))

    # initialise an icosphere
    ico = Icosphere(level=7)

    # pick 5 random images to visualise
    # np.random.shuffle(paths)
    # paths = paths[:5]
    paths = [paths[0]]

    # loop through the paths and visualise the images
    for path in tqdm(paths, total=len(paths), desc="[INFO] plotting the images: "):
        # read in the data and extract the images and labels
        arr = np.load(path)
        imgs, lbls = arr["data"], arr["labels"]

        # plot the image and save it to disk
        output_img_path = f"figs/area_1/{path.split(os.path.sep)[-1].split('.')[0]}_lvl7.html"
        mp.plot(ico.vertices, ico.faces, c=imgs[:ico.vertices.shape[0], :3], filename=output_img_path)

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

    for face in tqdm(ico.faces, total=ico.faces.shape[0]):
        # get the coordinates for the vertices of the current face
        coords = ico.vertices[face]

        # compute the coordinates of the midpoint vertices
        mid = np.vstack([[coords[x, :]] for x in [[0, 1], [1, 2], [0, 2]]]).mean(axis=1)

        # normalize the midpoint coordinates
        mid = normalize(mid)[:, None, :]

        # get the indices of the midpoint vertices
        idx1 = np.argmax((1 * np.equal(ico_up.vertices, mid[0])).sum(axis=1))
        idx2 = np.argmax((1 * np.equal(ico_up.vertices, mid[1])).sum(axis=1))
        idx3 = np.argmax((1 * np.equal(ico_up.vertices, mid[2])).sum(axis=1))

        # bilinear interpolation
        img[idx1] = (img[face[0]] + img[face[1]]) / 2
        img[idx2] = (img[face[1]] + img[face[2]]) / 2
        img[idx3] = (img[face[0]] + img[face[2]]) / 2

    # plot the image and save it to disk
    output_img_path = f"figs/area_1/{paths[0].split(os.path.sep)[-1].split('.')[0]}_lvl7.html"
    mp.plot(ico_up.vertices, ico_up.faces, c=img[:ico_up.vertices.shape[0], :3], filename=output_img_path)
