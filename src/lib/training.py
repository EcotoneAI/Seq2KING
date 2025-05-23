
from .models import BaseTransformer
from .params import *
from .saveload import load_checkpoint, save_checkpoint

import os
from timeit import default_timer

import torch
import onnx
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

# Types
from collections.abc import Callable


tok_loss = nn.MSELoss()
def tokenized_masked_loss(input, target, padval):
    mask = torch.where(target != padval, True, False)
    # Use this to reduce values included in the loss calculation
    input, target = torch.masked_select(input, mask), torch.masked_select(target, mask)
    return tok_loss(input, target)

def train_model_tokenized(model: BaseTransformer,
                optimizer: torch.optim.Optimizer,
                train_data: DataLoader,
                validate_data: DataLoader,
                output_run_dir: str,
                machine: str,
                loss_fcn: Callable,
                padval: float,
                epochs = 15,
                checkpoint_at = -1,
                load = True,
                batch_pr = 200,
                writer: SummaryWriter = None,
                output_onnx=True,
                **kwargs):
    # TODO: combine with train_model, better way to pass different output shapes
    mname = model.get_name()
    start_epoch = -1

    print(f"Training {mname}")

    # Attempt to load the previous checkpoint
    if load:
        checkpoint, statedict = load_checkpoint(mname, machine, output_run_dir)
        if checkpoint and statedict:
            start_epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["opt_state_dict"])
            model.load_state_dict(statedict)
        else:
            print("No checkpoint found to load. Using base model")


    # Save basic hyperparams
    # if writer:
    #     writer.add_hparams(model.get_hyperparameters(True), {})
    #     writer.flush()

    # Helps somehow? https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    torch.backends.cudnn.benchmark = True

    loss_arr = []
    validation_arr = []
    time_arr = []
    for e in range(start_epoch+1, start_epoch+1 + epochs ):
        model.train()
        running_loss = 0.0
        running_time = 0.0

        for i, (data, data_y, mask) in enumerate(train_data, 0):
            # data = data.to(device) # Should already be on the device?

            # Write out a view of the NN graph, just once
            if e == 0 and i == 0:
                # if writer:
                #     writer.add_graph(model, data)
                #     writer.flush()
                if output_onnx:
                    torch.onnx.export(model, (data, mask), os.path.join(output_run_dir, f'{mname}_model.onnx'), input_names=["src_seq", "mask"])

            start_time = default_timer()
            # zero the parameter gradients
            # Supposedly less memory operations with set_to_none=True?
            optimizer.zero_grad(set_to_none=True)
            # forward + backward + optimize
            out_seq = model(data, key_padding_mask=mask)
            # Loss function
            if padval is None:
                loss = loss_fcn(out_seq, data_y)
            else:
                loss = loss_fcn(out_seq, data_y, padval)
            loss.backward()
            optimizer.step()
            running_time = default_timer() - start_time

            running_loss += loss.item()
            # Print and save statistics, every batch_pr amt of data
            if i % batch_pr == batch_pr - 1:
                avg_loss = running_loss / batch_pr
                loss_arr.append(avg_loss)
                avg_time = running_time / batch_pr
                time_arr.append(avg_time)

                # Determine validation loss
                model.eval()
                model.train(False)
                v_arr = []
                with torch.no_grad():
                    for v_data, v_y, v_mask in validate_data:
                        v_data = v_data.to(device)
                        v_mask = v_mask.to(device)
                        out_seq_v = model(v_data, v_mask)
                        if padval is None:
                            v_arr.append(loss_fcn(out_seq_v, v_y.to(device)).item())
                        else:
                            v_arr.append(loss_fcn(out_seq_v, v_y.to(device), padval).item())
                validation_arr.append(np.mean(v_arr))
                model.train(True)

                # Write out stats
                print(f"[{e}, {i+1}] loss: {avg_loss}, validation loss: {validation_arr[-1]}, average train time (sec): {avg_time}")
                if writer:
                    writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : validation_arr[-1] },
                            e * len(train_data) + i)
                    writer.add_scalar('Average Train Time (s)', avg_time, e * len(train_data) + i)
                    writer.flush()

                running_loss = 0.0
                running_time = 0.0

        # Save output to checkpoint dict
        if e % checkpoint_at == checkpoint_at - 1:
            save = Checkpoint(mname, e, loss_arr, validation_arr, optimizer.state_dict(), time_arr)
            save_checkpoint(model, machine, output_run_dir, save)
            loss_arr = []
            validation_arr = []
            time_arr = []

    # Always save output at end
    save = Checkpoint(mname, e, loss_arr, validation_arr, optimizer.state_dict(), time_arr)
    save_checkpoint(model, machine, output_run_dir, save)

    print('Finished Training')


def train_model(model: BaseTransformer,
                optimizer: torch.optim.Optimizer,
                train_data: DataLoader,
                validate_data: DataLoader,
                output_run_dir: str,
                machine: str,
                loss_fcn: Callable,
                epochs = 15,
                checkpoint_at = -1,
                load = True,
                batch_pr = 200,
                writer: SummaryWriter = None,
                output_onnx=True,
                with_masked=False,
                **kwargs):
    mname = model.get_name()
    start_epoch = -1

    print(f"Training {mname}")

    # Attempt to load the previous checkpoint
    if load:
        checkpoint, statedict = load_checkpoint(mname, machine, output_run_dir)
        if checkpoint and statedict:
            start_epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["opt_state_dict"])
            model.load_state_dict(statedict)
        else:
            print("No checkpoint found to load. Using base model")


    # Save basic hyperparams
    # if writer:
    #     writer.add_hparams(model.get_hyperparameters(True), {})
    #     writer.flush()

    # Helps somehow? https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    torch.backends.cudnn.benchmark = True

    loss_arr = []
    validation_arr = []
    time_arr = []
    for e in range(start_epoch+1, start_epoch+1 + epochs ):
        model.train()
        running_loss = 0.0
        running_time = 0.0

        for i, data in enumerate(train_data, 0):
            # data = data.to(device) # Should already be on the device?

            if with_masked:
                data, data_y = data
            else:
                data_y = data

            # Write out a view of the NN graph, just once
            if e == 0 and i == 0:
                # if writer:
                #     writer.add_graph(model, data)
                #     writer.flush()
                if output_onnx:
                    torch.onnx.export(model, data, os.path.join(output_run_dir, f'{mname}_model.onnx'), input_names=["src_seq"])

            start_time = default_timer()
            # zero the parameter gradients
            # Supposedly less memory operations with set_to_none=True?
            optimizer.zero_grad(set_to_none=True)
            # forward + backward + optimize
            out_seq = model(data)
            # Loss function
            loss = loss_fcn(out_seq, data_y)
            loss.backward()
            optimizer.step()
            running_time = default_timer() - start_time

            running_loss += loss.item()
            # Print and save statistics, every batch_pr amt of data
            if i % batch_pr == batch_pr - 1:
                avg_loss = running_loss / batch_pr
                loss_arr.append(avg_loss)
                avg_time = running_time / batch_pr
                time_arr.append(avg_time)

                # Determine validation loss
                model.eval()
                model.train(False)
                v_arr = []
                with torch.no_grad():
                    for v_data in validate_data:
                        if with_masked:
                            v_data, v_y = v_data
                        else:
                            v_y = v_data
                        v_data = v_data.to(device)
                        out_seq_v = model(v_data)
                        v_arr.append(loss_fcn(out_seq_v, v_y.to(device)).item())
                validation_arr.append(np.mean(v_arr))
                model.train(True)

                # Write out stats
                print(f"[{e}, {i+1}] loss: {avg_loss}, validation loss: {validation_arr[-1]}, average train time (sec): {avg_time}")
                if writer:
                    writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : validation_arr[-1] },
                            e * len(train_data) + i)
                    writer.add_scalar('Average Train Time (s)', avg_time, e * len(train_data) + i)
                    writer.flush()

                running_loss = 0.0
                running_time = 0.0

        # Save output to checkpoint dict
        if e % checkpoint_at == checkpoint_at - 1:
            save = Checkpoint(mname, e, loss_arr, validation_arr, optimizer.state_dict(), time_arr)
            save_checkpoint(model, machine, output_run_dir, save)
            loss_arr = []
            validation_arr = []
            time_arr = []

    # Always save output at end
    save = Checkpoint(mname, e, loss_arr, validation_arr, optimizer.state_dict(), time_arr)
    save_checkpoint(model, machine, output_run_dir, save)

    print('Finished Training')
