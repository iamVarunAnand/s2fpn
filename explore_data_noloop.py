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
    level = 4
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

    # get the coordinates for the vertices of the current face
    coords = ico.vertices[ico.faces]

    # compute the coordinates of the midpoint vertices
    mid1 = coords[:, [0,1], :].mean(axis=1)
    mid2 = coords[:, [1,2], :].mean(axis=1)
    mid3 = coords[:, [2,0], :].mean(axis=1)

    # normalize the midpoint coordinates
    mid1 = normalize(mid1)
    mid2 = normalize(mid2)
    mid3 = normalize(mid3)

    # get the indices of the midpoint vertices
    mid1_mask = ((1 * np.equal(ico_up.vertices, mid1[:, None])).sum(axis=2)) == 3
    idx1 = np.argmax(mid1_mask, axis=1)

    mid2_mask = ((1 * np.equal(ico_up.vertices, mid2[:, None])).sum(axis=2)) == 3
    idx2 = np.argmax(mid2_mask, axis=1)



    mid3_mask = ((1 * np.equal(ico_up.vertices, mid3[:, None])).sum(axis=2)) == 3
    idx3 = np.argmax(mid3_mask, axis=1)


    # bilinear interpolation

    av1 = (img[ico.faces[:,0]] + img[ico.faces[:,1]]) / 2
    av2 = (img[ico.faces[:,1]] + img[ico.faces[:,2]]) / 2
    av3 = (img[ico.faces[:,0]] + img[ico.faces[:,2]]) / 2


    img[idx1] = av1
    img[idx2] = av2
    img[idx3] = av3

    print(img)

    # plot the image and save it to disk
    output_img_path = f"figs/area_1/{paths[0].split(os.path.sep)[-1].split('.')[0]}_upsample_test.html"
    mp.plot(ico_up.vertices, ico_up.faces, c=img[:ico_up.vertices.shape[0], :3], filename=output_img_path)

    output_img_path = f"figs/area_1/{paths[0].split(os.path.sep)[-1].split('.')[0]}_test.html"
    mp.plot(ico.vertices, ico.faces, c=img[:ico.vertices.shape[0], :3], filename=output_img_path)

