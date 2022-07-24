# import the necessary packages
from uscnn.models import SphericalUNet, SphericalFPNet
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from collections import OrderedDict
from data import ClimateSegLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import logging
import shutil
import torch
import wandb
import os

# class map - mapping label indices to class name
class_map = {
    0: "background",
    1: "tropical_cyclones",
    2: "atmospheric_river"
}


def save_checkpoint(state, is_best, epoch, output_folder, filename, logger):
    if epoch > 1:
        os.remove(output_folder + filename + '_%03d' % (epoch - 1) + '.pth.tar')
    torch.save(state, output_folder + filename + '_%03d' % epoch + '.pth.tar')
    if is_best:
        logger.info("Saving new best model")
        shutil.copyfile(output_folder + filename + '_%03d' % epoch + '.pth.tar',
                        output_folder + filename + '_best.pth.tar')


def iou_score(pred_cls, true_cls, nclass=3):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    iou = []
    for i in range(nclass):
        intersect = ((pred_cls == i).to(torch.int32) + (true_cls == i).to(torch.int32)).eq(2).sum().item()
        union = ((pred_cls == i).to(torch.int32) + (true_cls == i).to(torch.int32)).ge(1).sum().item()
        iou_ = intersect / union
        iou.append(iou_)
    return np.array(iou)


def average_precision(score_cls, true_cls, nclass=3):
    score = score_cls.cpu().numpy()
    true = label_binarize(true_cls.cpu().numpy().reshape(-1), classes=[0, 1, 2])
    score = np.swapaxes(score, 1, 2).reshape(-1, nclass)
    return average_precision_score(true, score)


def accuracy(pred_cls, true_cls, nclass=3):
    """
    compute per-node classification accuracy
    """
    accu = []
    for i in range(nclass):
        intersect = ((pred_cls == i).to(torch.int32) + (true_cls == i).to(torch.int32)).eq(2).sum().item()
        thiscls = (true_cls == i).to(torch.int32).sum().item()
        accu.append(intersect / thiscls)
    return np.array(accu)


def train(args, model, train_loader, optimizer, epoch, device, logger):
    if args.balance:
        w = torch.tensor([0.00102182, 0.95426438, 0.04471379]).to(device)
    else:
        w = torch.tensor([1.0, 1.0, 1.0]).to(device)

    model.train()

    tot_loss = 0
    count = 0

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)

        loss = F.cross_entropy(output, target, weight=w)
        loss.backward()

        optimizer.step()

        tot_loss += loss.item()
        count += data.size()[0]

        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    tot_loss /= count

    return tot_loss


def test(args, model, test_loader, device, logger):
    if args.balance:
        w = torch.tensor([0.00102182, 0.95426438, 0.04471379]).to(device)
    else:
        w = torch.tensor([1.0, 1.0, 1.0]).to(device)

    model.eval()

    test_loss, aps, count = 0, 0, 0
    ious, accus = np.zeros(3), np.zeros(3)

    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader)):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            pred = output.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability

            test_loss += F.cross_entropy(output, target, weight=w).item()  # sum up batch loss

            iou = iou_score(pred, target, nclass=3)
            ap = average_precision(output, target)
            accu = accuracy(pred, target, nclass=3)

            n_data = data.size()[0]

            ious += iou * n_data
            aps += ap * n_data
            accus += accu * n_data
            count += n_data

    ious /= count
    aps /= count
    accus /= count
    test_loss /= count

    logger.info(
        f'[INFO] test set - loss: {test_loss}; mAP: {aps:.4f}; mIoU: {np.mean(ious):.4f}; accuracy: {accus[0]:.4f}, {accus[1]:.4f}, {accus[2]:.4f}; IoU: {ious[0]:.4f}, {ious[1]:.4f}, {ious[2]:.4f}')

    return test_loss, accus, aps, ious


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Climate Segmentation')
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
    parser.add_argument('--feat', type=int, default=16, help='filter dimensions')
    parser.add_argument("--downsample", type=str, default='drop',
                        choices=['drop', 'average'], help="downsampling method to use")
    parser.add_argument('--upsample', type=str, default='zero-pad',
                        choices=['zero-pad', 'bilinear'], help='upsampling method to use')
    parser.add_argument('--log_dir', type=str, default="log", help='log directory for run')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--resume', type=str, default=None, help="path to checkpoint if resume is needed")
    parser.add_argument('--balance', action="store_true", help="switch for label frequency balancing")
    args = parser.parse_args()

    # set the appropriate training device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # create the log directory, if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # setup the logger
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
    wandb.init(project='climatenet', entity='tomvarun', config=config)

    # rename the weights and biases run
    if args.model == "fpn":
        wandb.run.name = f"climate fpn l{args.min_level}:{args.max_level} - {args.downsample} - {args.upsample}"
    else:
        wandb.run.name = f"climate unet - {args.downsample} - {args.upsample}"

    # set the experiment seed
    torch.manual_seed(args.seed)

    # initialise the train and validation sets
    trainset = ClimateSegLoader(args.data_folder, "train")
    testset = ClimateSegLoader(args.data_folder, "test")

    # initialise the data loaders
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=2)

    # initialise the model
    if args.model == "fpn":
        model = SphericalFPNet(in_ch=16, out_ch=3, max_level=args.max_level, min_level=args.min_level,
                               fdim=args.feat, downsample=args.downsample, upsample=args.upsample)
    elif args.model == "unet":
        model = SphericalUNet(in_ch=16, out_ch=3, max_level=args.max_level, min_level=args.min_level,
                              fdim=args.feat, downsample=args.downsample, upsample=args.upsample)
    else:
        print("Model Not Recognised")

    model = nn.DataParallel(model)
    model.to(device)

    if args.resume:
        resume_dict = torch.load(args.resume)

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

        load_my_state_dict(model, resume_dict)

    logger.info("{} parameters in total".format(sum(x.numel() for x in model.parameters())))

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)

    checkpoint_path = os.path.join(args.log_dir, 'checkpoint_latest.pth.tar')

    # loop through the epochs and train the model
    best_ap = 0
    for epoch in range(1, args.epochs + 1):
        if args.decay:
            scheduler.step(epoch)

        # train loop
        trn_loss = train(args, model, train_loader, optimizer, epoch, device, logger)

        # test loop
        tst_loss, tst_accs, tst_ap, tst_ious = test(args, model, test_loader, device, logger)

        # check to see if a new best was reached (in terms of average precision)
        is_best = tst_ap > best_ap
        if is_best:
            best_ap = tst_ap

        # log metrics to weights and biases
        wandb.log({
            "epoch": epoch,
            "trn_loss": trn_loss,
            "tst_loss": tst_loss,
            f"mAP": tst_ap,
            f"mIoU": np.mean(tst_ious),
            f"{class_map[0]}_acc": tst_accs[0],
            f"{class_map[1]}_acc": tst_accs[1],
            f"{class_map[2]}_acc": tst_accs[2],
            f"{class_map[0]}_iou": tst_ious[0],
            f"{class_map[1]}_iou": tst_ious[1],
            f"{class_map[2]}_iou": tst_ious[2]
        })

        # remove sparse matrices since they cannot be stored
        state_dict_no_sparse = [it for it in model.state_dict().items() if it[1].type() !=
                                "torch.cuda.sparse.FloatTensor"]
        state_dict_no_sparse = OrderedDict(state_dict_no_sparse)

        # save the model to disk
        save_checkpoint({
            'epoch': epoch,
            'state_dict': state_dict_no_sparse,
            'best_ap': best_ap,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, checkpoint_path, "_UNet", logger)


if __name__ == "__main__":
    main()
