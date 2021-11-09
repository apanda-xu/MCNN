import torch
import argparse

def set_value(new_value, default_value):
    if new_value is None:
        return default_value
    else:
        return new_value

def config():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="MCNN")
    parser.add_argument("--dataset", type=str, default="SHHA")
    parser.add_argument("--rand_seed", type=int, default=64678)
    parser.add_argument("--test_mode", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--parallel", type=bool, default=False)
    parser.add_argument("--use_tensorboard", type=bool, default=True) 
    parser.add_argument("--workspace", type=str, default="./")
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--logdir", type=str, default="log") 
    parser.add_argument("--device", type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--train_path", type= str, default=None)
    parser.add_argument("--train_gt_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--val_gt_path", type=str, default=None)   
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--test_gt_path", type=str, default=None) 
    args = parser.parse_args()

    # set values for arguments
    if args.dataset == "SHHA":
        args.lr                 = set_value(args.lr, 0.00001)
        args.max_epoch          = set_value(args.max_epoch, 2000)
        args.train_batch_size   = set_value(args.train_batch_size, 1)
        args.test_batch_size    = set_value(args.test_batch_size, 1)
        args.train_path         = set_value(args.train_path, "./data/formatted_trainval/shanghaitech_part_A_patches_9/train")
        args.train_gt_path      = set_value(args.train_gt_path, "./data/formatted_trainval/shanghaitech_part_A_patches_9/train_den")
        args.val_path           = set_value(args.val_path, "./data/formatted_trainval/shanghaitech_part_A_patches_9/val")
        args.val_gt_path        = set_value(args.val_gt_path, "./data/formatted_trainval/shanghaitech_part_A_patches_9/val_den")
        args.test_path          = set_value(args.test_path, './data/original/shanghaitech/part_A_final/test_data/images/')
        args.test_gt_path       = set_value(args.test_gt_path, './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/')

    elif args.dataset == "SHHB":
        args.lr                 = set_value(args.lr, 0.00001)
        args.max_epoch          = set_value(args.max_epoch, 2000)
        args.train_batch_size   = set_value(args.train_batch_size, 1)
        args.test_batch_size    = set_value(args.test_batch_size, 1)
        args.train_path         = set_value(args.train_path, "./data/formatted_trainval/shanghaitech_part_B_patches_9/train")
        args.train_gt_path      = set_value(args.train_gt_path, "./data/formatted_trainval/shanghaitech_part_B_patches_9/train_den")
        args.val_path           = set_value(args.val_path, "./data/formatted_trainval/shanghaitech_part_B_patches_9/val")
        args.val_gt_path        = set_value(args.val_gt_path, "./data/formatted_trainval/shanghaitech_part_B_patches_9/val_den")
        args.test_path          = set_value(args.test_path, './data/original/shanghaitech/part_B_final/test_data/images/')
        args.test_gt_path       = set_value(args.test_gt_path, './data/original/shanghaitech/part_B_final/test_data/ground_truth_csv/')

    else:
        print("%s is not support!\n" % args.dataset)
        exit()

    return args