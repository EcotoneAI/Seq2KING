import torch
from collections import namedtuple

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

use_cuda = device == "cuda"

## FORMAT: {machine}, {model}, {datetime}
state_dict_str = "{}_{}_state-dict_{}.pt"
state_dict_notime = "{}_{}_state-dict"
checkpoint_dict_str = "{}_{}_checkpoint_{}.tar"
checkpoint_dict_notime = "{}_{}_checkpoint"

Checkpoint = namedtuple("Checkpoint", ["model", "epoch", "loss", "validation_loss", "opt_state_dict", "train_time"])
