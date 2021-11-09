import os
import torch
import torch.nn as nn
import torch.optim as optim

from configs.default import config
from tools.helper import list_args

# ---------------cfg---------------
args = config()
args.workspace = os.getcwd()
seed = args.rand_seed
dataset = args.dataset
net = args.net
args.test_mode = True
list_args(args)

# ---------------environments---------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ---------------dataloader---------------
if dataset in ["SHHA", "SHHB"]:
    from datasets.shha import DataLoader
else:
    exit()

# ---------------trainer---------------
if net in ["MCNN"]:
    from tools.trainer import trainer

# ---------------start training---------------
trainer = trainer(args, DataLoader)
trainer.test()
