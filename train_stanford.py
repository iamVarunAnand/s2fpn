# import the necessary packages
from uscnn.models import SphericalUNet, SphericalFPNet
from torch.utils.data import DataLoader
from collections import OrderedDict
from data import S2D3DSegLoader
from tabulate import tabulate
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import logging
import shutil
import torch
import wandb
import json
import os

# # stop pytorch from caching GPU memory
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


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

# initialise metadata
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
# label_weight = 1/np.array(label_ratio)/np.sum((1/np.array(label_ratio))[keep])
label_weight = 1 / np.log(1.02 + np.array(label_ratio))
label_weight[drop] = 0
label_weight = label_weight.astype(np.float32)


def getmem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a

    return ([t, r, a, f])


def save_checkpoint(state, is_best, epoch, output_folder, filename, logger):
    if epoch > 1:
        os.remove(output_folder + filename + '_%03d' % (epoch - 1) + '.pth.tar')

    torch.save(state, output_folder + filename + '_%03d' % epoch + '.pth.tar')

    if is_best:
        logger.info("Saving new best model")
        shutil.copyfile(output_folder + filename + '_%03d' % epoch + '.pth.tar',
                        output_folder + filename + '_best.pth.tar')


def iou_score(pred_cls, true_cls, nclass=15, drop=drop):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """

    intersect_ = []
    union_ = []

    for i in range(nclass):
        if i not in drop:
            intersect = ((pred_cls == i).to(torch.int32) + (true_cls == i).to(torch.int32)).eq(2).sum().item()
            union = ((pred_cls == i).to(torch.int32) + (true_cls == i).to(torch.int32)).ge(1).sum().item()
            intersect_.append(intersect)
            union_.append(union)

    return np.array(intersect_), np.array(union_)


def accuracy(pred_cls, true_cls, nclass=15, drop=drop):
    positive = torch.histc(true_cls.cpu().float(), bins=nclass, min=0, max=nclass, out=None)

    per_cls_counts = []
    tpos = []

    for i in range(nclass):
        if i not in drop:
            true_positive = ((pred_cls == i).to(torch.int32) + (true_cls == i).to(torch.int32)).eq(2).sum().item()
            tpos.append(true_positive)
            per_cls_counts.append(positive[i])

    return np.array(tpos), np.array(per_cls_counts)


def train(args, model, train_loader, optimizer, epoch, device, logger, keep_id=None):
    w = torch.tensor(label_weight).to(device)

    model.train()

    tot_loss = 0
    count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if keep_id is not None:
            output = output[:, :, keep_id]
            target = target[:, keep_id]

        loss = F.cross_entropy(output, target.long(), weight=w)

        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        count += data.size()[0]

        if batch_idx % args.log_interval == 0:
            # print(f"\nMemory Total, Reserved, Allocated and Free: {getmem()}")
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    tot_loss /= count

    return tot_loss


def test(args, model, test_loader, epoch, device, logger, keep_id=None):
    w = torch.tensor(label_weight).to(device)
    model.eval()

    test_loss = 0

    ints_ = np.zeros(len(classes) - len(drop))
    unis_ = np.zeros(len(classes) - len(drop))
    per_cls_counts = np.zeros(len(classes) - len(drop))
    accs = np.zeros(len(classes) - len(drop))

    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            n_data = data.size()[0]

            if keep_id is not None:
                output = output[:, :, keep_id]
                target = target[:, keep_id]

            loss = F.cross_entropy(output, target.long(), weight=w).item()  # sum up batch loss
            test_loss += loss

            pred = (output.max(dim=1, keepdim=False)[1]).to(torch.int32)  # get the index of the max log-probability
            int_, uni_ = iou_score(pred, target)
            tpos, pcc = accuracy(pred, target)
            ints_ += int_
            unis_ += uni_
            accs += tpos
            per_cls_counts += pcc
            count += n_data

    ious = ints_ / unis_
    accs /= per_cls_counts
    test_loss /= count

    logger.info('[Epoch {} {} stats]: MIoU: {:.4f}; Mean Accuracy: {:.4f}; Avg loss: {:.4f}'.format(
        epoch, test_loader.dataset.partition, np.mean(ious), np.mean(accs), test_loss))

    # tabulate mean iou
    logger.info(tabulate(dict(zip(class_names[1:-1], [[iou] for iou in ious])), headers="keys"))

    # tabulate mean acc
    logger.info(tabulate(dict(zip(class_names[1:-1], [[acc] for acc in accs])), headers="keys"))

    return np.mean(np.mean(ious)), accs, ious, test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Indoor Segmentation')
    parser.add_argument('--Name', type=str, default="NoName",
                        help='Run Name')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_folder', type=str, default="processed_data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--max_level', type=int, default=7, help='max mesh level')
    parser.add_argument('--min_level', type=int, default=0, help='min mesh level')
    parser.add_argument('--model', type=str, choices=["fpn", "unet"], default="fpn")
    parser.add_argument('--feat', type=int, default=4, help='filter dimensions')
    parser.add_argument("--downsample", type=str, default='drop',
                        choices=['drop', 'average'], help="downsampling method to use")
    parser.add_argument('--upsample', type=str, default='zero-pad',
                        choices=['zero-pad', 'bilinear'], help='upsampling method to use')
    parser.add_argument('--log_dir', type=str, default="log",
                        help='log directory for run')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--resume', type=str, default=None, help="path to checkpoint if resume is needed")
    parser.add_argument('--fold', type=int, choices=[1, 2, 3],
                        required=True, help="choice among 3 fold for cross-validation")
    parser.add_argument('--blackout_id', type=str, default="", help="path to file storing blackout_id")
    parser.add_argument('--in_ch', type=str, default="rgbd", choices=["rgb", "rgbd"], help="input channels")
    parser.add_argument('--train_stats_freq', default=0, type=int,
                        help="frequency for printing training set stats. 0 for never.")
    args = parser.parse_args()

    # set the appropriate training device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print(f"[INFO] total memory, reserved, allocated, free: {getmem()}")

    # logger and snapshot current code
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(args))

    # initialise Weights and Biases run
    config = {"Batch Size": args.batch_size, "Epochs": args.epochs, "LR": args.lr, "fdim": args.feat}
    wandb.init(project='SCNN', entity='tomvarun', config=config)

    # rename the weights and biases run
    if args.model == "fpn":
        wandb.run.name = f"ugscnn fpn l{args.min_level}:{args.max_level} - {args.downsample} - {args.upsample} - fold {args.fold}"
    else:
        wandb.run.name = f"ugscnn unet - {args.downsample} - {args.upsample} - fold {args.fold}"

    trainset = S2D3DSegLoader(args.data_folder, "train", fold=args.fold, sp_level=args.max_level, in_ch=len(args.in_ch))
    valset = S2D3DSegLoader(args.data_folder, "test", fold=args.fold, sp_level=args.max_level, in_ch=len(args.in_ch))
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # load model
    if args.model == "fpn":
        model = SphericalFPNet(in_ch=len(args.in_ch), out_ch=len(
            classes), max_level=args.max_level, min_level=args.min_level, fdim=args.feat, downsample=args.downsample, upsample=args.upsample)
    elif args.model == "unet":
        model = SphericalUNet(in_ch=len(args.in_ch), out_ch=len(
            classes), max_level=args.max_level, min_level=args.min_level, fdim=args.feat, downsample=args.downsample, upsample=args.upsample)
    else:
        print("Model Not Recognised")

    # initialise bias of out conv
    dist = json.load(open("data/bias_init.json", "r"))
    model.out_conv.bias.data = torch.Tensor(list(dist.values()))

    model = nn.DataParallel(model)
    model.to(device)

    if args.blackout_id:
        blackout_id = np.load(args.blackout_id)
        keep_id = np.argwhere(np.isin(np.arange(model.module.nv_max), blackout_id, invert=True))
    else:
        keep_id = None

    start_ep = 0
    best_miou = 0
    if args.resume:
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict['epoch']
        best_miou = resume_dict['best_miou']

        def load_my_state_dict(self, state_dict, exclude='none'):
            from torch.nn.parameter import Parameter

            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                if exclude in name:
                    continue
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)

        load_my_state_dict(model, resume_dict['state_dict'])

    logger.info("{} parameters in total".format(sum(x.numel() for x in model.parameters())))

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    checkpoint_path = os.path.join(args.log_dir, 'checkpoint_latest.pth.tar')

    # training loop
    for epoch in range(start_ep + 1, args.epochs + 1):
        if args.decay:
            scheduler.step(epoch)

        loss = train(args, model, train_loader, optimizer, epoch, device, logger, keep_id)
        miou, accs, ious, val_loss = test(args, model, val_loader, epoch, device, logger, keep_id)

        wandb.log({"Train Loss": loss,
                   "Val Loss": val_loss,
                   f"{class_names[1]} Acc": accs[0],
                   f"{class_names[2]} Acc": accs[1],
                   f"{class_names[3]} Acc": accs[2],
                   f"{class_names[4]} Acc": accs[3],
                   f"{class_names[5]} Acc": accs[4],
                   f"{class_names[6]} Acc": accs[5],
                   f"{class_names[7]} Acc": accs[6],
                   f"{class_names[8]} Acc": accs[7],
                   f"{class_names[9]} Acc": accs[8],
                   f"{class_names[10]} Acc": accs[9],
                   f"{class_names[11]} Acc": accs[10],
                   f"{class_names[12]} Acc": accs[11],
                   f"{class_names[13]} Acc": accs[12],
                   "Mean Acc": np.mean(accs),
                   f"{class_names[1]} IoU": ious[0],
                   f"{class_names[2]} IoU": ious[1],
                   f"{class_names[3]} IoU": ious[2],
                   f"{class_names[4]} IoU": ious[3],
                   f"{class_names[5]} IoU": ious[4],
                   f"{class_names[6]} IoU": ious[5],
                   f"{class_names[7]} IoU": ious[6],
                   f"{class_names[8]} IoU": ious[7],
                   f"{class_names[9]} IoU": ious[8],
                   f"{class_names[10]} IoU": ious[9],
                   f"{class_names[11]} IoU": ious[10],
                   f"{class_names[12]} IoU": ious[11],
                   f"{class_names[13]} IoU": ious[12],
                   "Mean IoU": np.mean(ious)})

        if args.train_stats_freq > 0 and (epoch % args.train_stats_freq == 0):
            _ = test(args, model, train_loader, epoch, device, logger, keep_id)
        if miou > best_miou:
            best_miou = miou
            is_best = True
        else:
            is_best = False

        # remove sparse matrices since they cannot be stored
        state_dict_no_sparse = [it for it in model.state_dict().items() if it[1].type() !=
                                "torch.cuda.sparse.FloatTensor"]
        state_dict_no_sparse = OrderedDict(state_dict_no_sparse)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': state_dict_no_sparse,
            'best_miou': best_miou,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, checkpoint_path, "_SUNet", logger)


if __name__ == "__main__":
    main()
