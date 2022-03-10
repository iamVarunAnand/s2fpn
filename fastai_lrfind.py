# import the necessary packages
from uscnn.models import SphericalFPNet
from data import S2D3DSegLoader
from fastai.learner import Learner
from fastai.vision.all import *
from fastai.data.core import *
from fastai.losses import *
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch

num_classes = 15
classes = [i for i in range(num_classes)]
class_names = ["unknown", "beam", "board", "bookcase", "ceiling", "chair", "clutter", "column",
               "door", "floor", "sofa", "table", "wall", "window", "invalid"]
drop = [0, 14]
keep = np.setdiff1d(classes, drop)
label_ratio = [0.04233976974675504, 0.014504436907968913, 0.017173225930738712,
               0.048004778186652164, 0.17384037404789865, 0.028626771620973622,
               0.087541966989014, 0.019508096683310605, 0.08321331842901526,
               0.17002664771895903, 0.002515611224467519, 0.020731298851232174,
               0.2625963729249342, 0.016994731594287146, 0.012382599143792165]

label_weight = 1 / np.log(1.02 + np.array(label_ratio))
label_weight[drop] = 0
label_weight = label_weight.astype(np.float32)


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)

class CustomLoss(nn.Module):
    def __init__(self, weight=None):
        super(CustomLoss, self).__init__()
        self.weight = weight
    
    def forward(self, inpt, trgt):
        return F.cross_entropy(inpt, trgt.long(), weight=self.weight)

class CustomLossFlat(BaseLoss):
    def __init__(self, *args, weight=None, **kwargs):
        super().__init__(CustomLoss, *args, weight=weight, **kwargs)
    
    def decodes(self, x): return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)


if __name__ == "__main__":
    # print(label_weight)
    train_ds = S2D3DSegLoader("../2d3ds_sphere", "train", 1, 5)
    val_ds = S2D3DSegLoader("../2d3ds_sphere", "test", 1, 5)

    train_dl = DataLoader(train_ds, bs=4, shuffle=True)
    val_dl = DataLoader(val_ds, bs=4, shuffle=False)

    dls = DataLoaders(train_dl, val_dl, device=torch.device('cuda'))

    model = SphericalFPNet(4, 15, fdim=32).cuda()

    w = torch.tensor(label_weight).cuda()
    learn = Learner(dls, model, loss_func=CustomLossFlat(weight=None, axis=1), opt_func=Adam)
    ret = learn.lr_find()

    print(ret)
    plt.savefig("lr_plot_adam.png")