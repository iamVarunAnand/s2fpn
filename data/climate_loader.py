# import the necessary packages
from torch.utils.data import Dataset
import numpy as np
import os

# precomputed mean and std of the dataset
precomp_mean = [26.160023, 0.98314494, 0.116573125, -0.45998842, 0.1930554, 0.010749293, 98356.03,
                100982.02, 216.13145, 258.9456, 3.765611e-08, 288.82578, 288.03925, 342.4827, 12031.449, 63.435772]
precomp_std = [17.04294, 8.164175, 5.6868863, 6.4967732, 5.4465833, 0.006383436, 7778.5957, 3846.1863,
               9.791707, 14.35133, 1.8771327e-07, 19.866386, 19.094095, 624.22406, 679.5602, 4.2283397]


class ClimateSegLoader(Dataset):
    """Data loader for Climate Segmentation dataset."""

    def __init__(self, data_dir, partition="train", normalize_mean=precomp_mean, normalize_std=precomp_std):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
        """
        assert(partition in ["train", "test", "val"])

        # get the filenames for the current split
        with open(os.path.join(data_dir, f"{partition}_split.txt"), "r") as f:
            lines = f.readlines()

        # build the paths to the files
        self.flist = [os.path.join(data_dir, line.replace('\n', '')) for line in lines]

        # initialise the dataset mean and std for normalization
        self.mean = np.expand_dims(normalize_mean, -1).astype(np.float32)
        self.std = np.expand_dims(normalize_std, -1).astype(np.float32)

    def __len__(self):
        # return the total number of files in the dataset
        return len(self.flist)

    def __getitem__(self, idx):
        # read the file at the current index
        f = np.load(self.flist[idx])

        # extract the data and the labels
        data = (f["data"] - self.mean) / self.std
        labels = f["labels"].astype(np.int32)

        # convert the labels from one-hot to categorical labels
        labels = np.argmax(labels, axis=0)

        # return the data and the labels
        return data, labels


if __name__ == "__main__":
    from tqdm import tqdm

    trn_dataset = ClimateSegLoader("../datasets/climate_sphere_l5/data_5_all", "train")
    val_dataset = ClimateSegLoader("../datasets/climate_sphere_l5/data_5_all", "val")
    tst_dataset = ClimateSegLoader("../datasets/climate_sphere_l5/data_5_all", "test")

    print(len(trn_dataset), len(val_dataset), len(tst_dataset))

    counts = np.array([0, 0, 0])
    for data, lbls in tqdm(iter(tst_dataset), total=len(tst_dataset)):
        unique, count = np.unique(lbls, return_counts=True)

        counts[unique] += count

    print(counts / counts.sum())
