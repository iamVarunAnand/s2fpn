# import the necessary packages
from uscnn.meshcnn import Icosphere
from tqdm import tqdm
import meshplot as mp
import numpy as np
import glob
import os

mp.offline()

np.random.seed(7)


if __name__ == "__main__":
    # grab the path to the dataset files
    paths = list(glob.glob("/home/varun/datasets/2d3ds_sphere/area_1/*.npz"))

    # sort the paths according to the split
    paths = sorted(paths, key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0].split("_")[-1]))

    # initialise an icosphere
    ico = Icosphere(level=7)

    # pick 5 random images to visualise
    np.random.shuffle(paths)
    paths = paths[:5]

    # loop through the paths and visualise the images
    for path in tqdm(paths, total=len(paths), desc="[INFO] plotting the images: "):
        # read in the data and extract the images and labels
        arr = np.load(path)
        imgs, lbls = arr["data"], arr["labels"]

        # plot the image and save it to disk
        output_img_path = f"figs/area_1/{path.split(os.path.sep)[-1].split('.')[0]}_lvl7_depth.html"
        mp.plot(ico.vertices, ico.faces, c=imgs[:ico.vertices.shape[0], -1], filename=output_img_path)
