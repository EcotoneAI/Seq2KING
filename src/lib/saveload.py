from .params import *
from .models import BaseTransformer

import os
import json
from datetime import datetime

import torch


def load_checkpoint(mname: str,
                    machine: str,
                    output_run_dir: str,
                    checkpoint_str: str = checkpoint_dict_notime,
                    statedict_str: str = state_dict_notime
                    ):
    """Load the given model's checkpoint dict and state params/dict

    Args:
        mname (str): model.get_name()
        machine (str): Usually "PC" or "Mac" or "RC"
        output_run_dir (str): an osPath
        checkpoint_str (str, optional): Defaults to checkpoint_dict_notime.
        statedict_str (str, optional): Defaults to state_dict_notime.

    Returns:
        checkpoint_dict, statedict ((dict, dict)):
    """
    checkpoint_dict, statedict = (None, None)

    if checkpoint_str:
        name_checkpoint = checkpoint_dict_notime.format(machine, mname)
        file_checkpoint = next((x for x in sorted(os.listdir(output_run_dir), reverse=True) if x.startswith(name_checkpoint)), None)
        # file_checkpoint = max(filter(lambda x: x.startswith(name_checkpoint)), os.listdir(output_run_dir), default=None)
        if file_checkpoint:
            print(f"Found checkpoint to load. Using: {file_checkpoint}")
            checkpoint_dict = torch.load(os.path.join(output_run_dir, file_checkpoint),
                                         map_location="cpu" if not use_cuda else None)

    if statedict_str:
        name_statedict = state_dict_notime.format(machine, mname)
        file_statedict = next((x for x in sorted(os.listdir(output_run_dir), reverse=True) if x.startswith(name_statedict)), None)
        if file_statedict:
            print(f"Found model state dict to load. Using: {file_statedict}")
            statedict = torch.load(os.path.join(output_run_dir, file_statedict),
                                   map_location="cpu" if not use_cuda else None)

    return checkpoint_dict, statedict


def save_checkpoint(model: BaseTransformer,
                    machine: str,
                    output_run_dir: str,
                    checkpoint_data: Checkpoint,
                    ):
    mname = model.get_name()
    dt = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    save = checkpoint_data._asdict()
    torch.save(save,
               os.path.join(output_run_dir,
                            checkpoint_dict_str.format(machine, mname, dt)))
    torch.save(model.state_dict(),
               os.path.join(output_run_dir,
                            state_dict_str.format(machine, mname, dt)))
    print(f"Saved checkpoint for epoch {checkpoint_data.epoch}: {machine}_{mname}")


def load_trained_model(ModelClass: BaseTransformer.__class__, machine, runname, model_str_name, dir):
    with open(os.path.join(dir, f"details_{runname}.json"), "r" ) as file:
        deets = json.load(file)

    deets[model_str_name].pop("desc", None)
    model = ModelClass(**deets[model_str_name]).to(device)

    _, statedict = load_checkpoint(model.get_name(), machine, output_run_dir=dir, checkpoint_str="")

    if not statedict:
        print(f"Could not find state dict for {model_str_name}")
        return None

    model.load_state_dict(statedict)

    return model
