"""train_1
The file that actually runs the training of models.
This is the initial test of running the transformer model, with:
- 4 layers encode/decode
- tanh activtion (relu cutting off vals?)
- slower optim


Author: Bhavana Jonnalagadda
"""

# Allow import from our custom lib python files
import sys
import os
import json

module_path = os.path.abspath(os.path.join('lib'))
if module_path not in sys.path:
    sys.path.append(module_path)

from timeit import default_timer
# Types
from collections.abc import Callable
from torch.utils.tensorboard import SummaryWriter

from lib.params import * # device, use_cuda, Checkpoint, various saving strs
from lib.datasets import KingMatrixDataset
from lib.models import BaseTransformer
from lib.saveload import *
from lib.training import train_model

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import numpy as np


def run():
    ### Global Parameters ###
    runname = "test2"
    machine = "Mac"
    datapath = "../Output/Heritage_UMAP/king_matrix.csv"
    outdir = os.path.join("../Output/Runs", runname)
    tensorboard_dir = "../Output/Tensorboard"
    SEED = 42

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    print(f"Using {device} device")
    ##########################


    ### Load Data ###
    dataset = KingMatrixDataset(datapath)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=(torch.cuda.is_available()), num_workers=4)
    print(len(dataset), dataset[0].shape, dataset[2502])
    #################


    ### Construct Model(s) ###
    model_name = "Test_2"
    run_details = dict(run_params=dict(
                        machine=machine,
                        epochs = 30,
                        checkpoint_at = 10,
                        load=False,
                        batch_pr = 500,
                        runname=runname
                        )
                    ) | {model_name: dict(
                            d_model=1,
                            num_head=1,
                            num_encoder_layers=3,
                            num_decoder_layers=3,
                            dim_feedforward=1028
                        )}
    model = BaseTransformer(**run_details[model_name]).to(device)
    print(model)

    # Save details
    with open(os.path.join(outdir, f"details_{runname}.json"), "w" ) as write:
        json.dump(run_details, write, indent=2 )
    ##########################


    ### Train Model(s) ###
    loss_fcn = nn.MSELoss()

    writer = SummaryWriter(os.path.join(tensorboard_dir, f'{machine}_{model.get_name()}_{runname}'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    # Set foreach=False to avoid OOM
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, foreach=False)

    train_model(model=model,
                optimizer=optimizer,
                train_data=train_dataloader,
                validate_data=train_dataloader,
                loss_fcn=loss_fcn,
                output_run_dir=outdir,
                machine=machine,
                epochs = 30,
                checkpoint_at = 10,
                load=False,
                batch_pr = 500,
                # **run_details["run_params"],
                writer=writer,
                output_onnx=True
            )

    writer.close()
    ######################

if __name__ == '__main__':
    run()
