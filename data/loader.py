# import the necessary packages
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import os

# sphere mesh size at different levels
nv_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]

# precomputed mean and std of the dataset
precomp_mean = [0.4974898, 0.47918808, 0.42809588, 1.0961773]
precomp_std = [0.23762763, 0.23354423, 0.23272438, 0.75536704]


class S2D3DSegLoader(Dataset):
    """
        Data loader for 2D3DS dataset.
    """

    def __init__(self, data_dir, partition, fold, sp_level, in_ch=4, normalize_mean=precomp_mean, normalize_std=precomp_std):
        """
            Args:
                data_dir: path to data directory
                partition: train or test
                fold: 1, 2 or 3 (for 3-fold cross-validation)
                sp_level: sphere mesh level. integer between 0 and 7.
        """

        # ensure passed in parameters are correct
        assert(partition in ["train", "test"])
        assert(fold in [1, 2, 3])

        # initialise the instance variables
        self.in_ch = in_ch
        self.nv = nv_sphere[sp_level]
        self.partition = partition

        # initialise the appropriate fold
        if fold == 1:
            train_areas = ['1', '2', '3', '4', '6']
            test_areas = ['5a', '5b']
        elif fold == 2:
            train_areas = ['1', '3', '5a', '5b', '6']
            test_areas = ['2', '4']
        elif fold == 3:
            train_areas = ['2', '4', '5a', '5b']
            test_areas = ['1', '3', '6']

        # initialise the appropriate split
        if partition == "train":
            self.areas = train_areas
        else:
            self.areas = test_areas

        # loop through the data files and build the file list
        self.flist = []
        for a in self.areas:
            # build the path to the folder and grab all files
            area_dir = os.path.join(data_dir, "area_" + a)
            self.flist += sorted(glob(os.path.join(area_dir, "*.npz")))

        # initialise the mean and standard deviation to be used for normalizing inputs
        self.mean = np.expand_dims(normalize_mean, -1).astype(np.float32)
        self.std = np.expand_dims(normalize_std, -1).astype(np.float32)

    def __len__(self):
        # compute and return the total size of the dataset
        return len(self.flist)

    def __getitem__(self, idx):
        # load the appropriate file
        arr = np.load(self.flist[idx])

        # extract the images and labels
        imgs = arr["data"].T[:self.in_ch, :self.nv].astype(np.float32)
        lbls = arr["labels"].T[:self.nv].astype(np.uint8)

        # normalize the images
        imgs = (imgs - self.mean) / self.std

        # return the loaded data
        return imgs, lbls
