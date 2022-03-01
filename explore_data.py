# import the necessary packages
from uscnn.utils import Icosphere
import uscnn.layers as layers
from tqdm import tqdm
import meshplot as mp
import numpy as np
import glob
import os

mp.offline()

np.random.seed(7)


if __name__ == "__main__":
    # grab the path to the dataset files
    paths = list(glob.glob(r"C:\Users\spike\Documents\Python Scripts\MLP_CW4\ugscnn-master\ugscnn-master\new\ugscnn\experiments\exp3_2d3ds\2d3ds_sphere\area_1\*.npz"))

    # sort the paths according to the split
    paths = sorted(paths, key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0].split("_")[-1]))

    # initialise an icosphere
    level = 0
    nv_pad = 30*(4**level)

    ico = Icosphere(level=level)
    n_faces = ico.faces.shape[0]
    print(n_faces)
    ico.subdivide()
    ico_up = Icosphere(level=level+1)
    # pick 5 random images to visualise
    np.random.shuffle(paths)
    paths = paths[:5]

    # loop through the paths and visualise the images
    for path in tqdm(paths, total=len(paths), desc="[INFO] plotting the images: "):
        # read in the data and extract the images and labels
        arr = np.load(path)
        imgs_raw, lbls = arr["data"], arr["labels"]

        imgs = imgs_raw[:ico.vertices.shape[0]]
        zeros_pad = np.zeros((nv_pad, 4))
        imgs = np.concatenate((imgs, zeros_pad))




        for face in ico_up.faces:
            a,b,c = face
            img(imgs[a]+imgs[b])/2

        for i in range(0, 1):
            imgs[ico.vertices.shape[0] + i, :] = np.array([[255],[0],[0],[0]])[:,-1]               #(imgs[i, :] + imgs[i + 1, :])/2
            imgs[ico.vertices.shape[0] + i + n_faces, :] = np.array([[0],[255],[0],[0]])[:,-1]     #(imgs[i + 1, :] + imgs[i + 2, :])/2       #
            imgs[ico.vertices.shape[0] + i + 2*n_faces, :] = np.array([[0],[0],[255],[0]])[:,-1]   #(imgs[i + 2, :] + imgs[i, :])/2       #

        # plot the image and save it to disk
        output_img_path = f"figs/area_1/{path.split(os.path.sep)[-1].split('.')[0]}_up.html"
        mp.plot(ico_up.vertices, c=imgs[:, :3], shading={"point_size": 0.2}, filename=output_img_path)

        output_img_path = f"figs/area_1/{path.split(os.path.sep)[-1].split('.')[0]}.html"
        mp.plot(ico.vertices,  c=imgs[:ico.vertices.shape[0], :3], shading={"point_size": 0.2}, filename=output_img_path)
