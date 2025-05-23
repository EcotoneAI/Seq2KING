import os
from collections.abc import Sequence, Callable

from .params import *

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch import nn
import pandas as pd

class KingMatrixDataset(Dataset):
    def __init__(self,
                 filepath: str,
                 no_zero=False,
                 pad=0,
                 long_dim_first=True):
        """Straight reads the file.

        Args:
            filepath (str): Where the king matrix is located, ends in .csv
            no_zero (bool, optional): Whether to reverse the seq and remove all zeros, or not. Defaults to False.
            pad (int, optional): how many zeros to pad the data vector with (for n_head divisibility)
            long_dim_first (bool, optional): Whether the 1n dim is second or first.
        """
        super().__init__()

        self.filename = filepath
        self.no_zero = no_zero
        self.long_dim_first = long_dim_first
        self.X = pd.read_csv(self.filename)
        self.X = self.X.fillna(0)
        # Add padding columns to beginning
        for i in range(pad):
            self.X.insert(0, "pad", 0, True)

        self._len = len(self.X)

    def __len__(self):
        return self._len


    def __getitem__(self, idx):
        if self.no_zero:
            vec = torch.tensor(self.X.iloc[idx, :][:(idx-1 if idx != 0 else None):(-1)],
                                dtype=torch.float,
                                device=device)
        else:
            vec = torch.tensor(self.X.iloc[idx, :],
                                dtype=torch.float,
                                device=device)

        # Add embedding dim
        return vec[:, None] if self.long_dim_first else vec[None, :]


class FilledMatrixDataset(KingMatrixDataset):
    """Flip the values along the diagonal, filling the redundant spots with the correct vals.
    Note: The first row and last column are not repeated across the matrix, which is why
        the diagonal starts at [1, 0] and ends at [-1, -2]. Inspect the original df X to see row/col names
    """
    def __init__(self,
                 filepath: str,
                 index_col=False,
                 pad=0,
                 long_dim_first=True,
                 mask: Callable = None,
                 normalize = False):
        super().__init__(filepath, False, pad, long_dim_first)

        self.mask = mask

        # Re-process X f there's an index col
        if index_col:
            self.X = pd.read_csv(self.filename, index_col=0)
            # Add padding columns to beginning
            for i in range(pad):
                self.X.insert(0, f"pad{i}", 0, True)

        self.X.iloc[1:, pad:-1] = np.rot90(np.fliplr(self.X.iloc[1:, pad:-1])) + self.X.iloc[1:, pad:-1]

        if normalize:
            minn = np.min(self.X.to_numpy())
            maxx = np.max(self.X.to_numpy())
            self.X = (self.X - minn) / (maxx - minn)

    def apply_mask(self, idx, mat: torch.Tensor):
        """For only long_dim = 0"""
        if self.mask is not None:
            inds, val = self.mask(mat.size(0), idx)
            mat_masked = mat.detach().clone()
            mat_masked[inds, :] = val
            return mat_masked, mat

        return mat


    def __getitem__(self, idx):
        vec = torch.tensor(self.X.iloc[idx, :],
                            dtype=torch.float,
                            device=device)

        # Add embedding dim
        # Get masked, then original mat if mask
        return self.apply_mask(idx, vec[:, None]) if self.long_dim_first else vec[None, :]


class TokenizedKingDataset(FilledMatrixDataset):
    """Input is a sequence of indices indicating which person/nodes, and the target/output to compare
    against is the resulting weights/kinship relationship matrix between those individuals. The
    output has a fixed max sequence length on the embedding dim, which is thus padded (and masked
    when comparing loss); the sequence length is obviously variable, but the non-padded target values
    is a square matrix. When batched, the sequence input/matrix output lengths themselves are padded
    out and masked with a PADDING_INDEX indice that the embedding layer knows about and is fixed.

    Args:
        filepath (str): Where the king matrix is located, ends in .csv
        index_col (bool, optional): If there's an index column in the csv. Defaults to False.
        filled (bool, optional): Whether to use filled matrix. Defaults to False.
        normalize (bool, optional): Whether to normalize vals to [0, 1]. Defaults to True.
        padval (_type_, optional): What to pad the target kinship weights matrix with. Defaults to np.nan.
        dsize (int, optional): The total size (n seqs) in the dataset. Defaults to 10000.
        maxseqlen (int, optional): The maximum length of a single sequence of indices; the datset
            makes sequences from length=2 up to maxseqlen. Defaults to 2502.
        maxind (int, optional): The maximum indice to use in the King matrix; indices up to
            maxind are used. Defaults to 2501.

    Raises:
        Exception: if input vars are bad in some way
    """
    def __init__(self,
                 filepath: str,
                 index_col=False,
                 filled=False,
                 normalize=True,
                 padval=-1,
                 dsize=10000,
                 maxseqlen=2502,
                 maxind=2501,
                 remove_starts=True):
        # long_dim_first doesn't matter since making 2 col matrix
        long_dim_first = False
        # No padding since only ever 2 cols
        pad = 0

        # Get filled matrix
        if filled:
            super().__init__(filepath, index_col, pad, long_dim_first=long_dim_first)
        # No filled matrix, just normal
        else:
            super(FilledMatrixDataset, self).__init__(filepath, no_zero=False, pad=pad, long_dim_first=long_dim_first)

        # Convert to numpy array
        self.X = self.X.to_numpy()

        if remove_starts:
            # Get rid of the row and col that don't contain reciprocal match
            # TODO: just add rowcol instead?!?!?
            self.X = self.X[1:, :-1]
        # Just add the correct col/row insteead. Note this is adding an extra ind to the rows that
        # will probably never be accessed
        else:
            if filled:
                self.X = np.concatenate((np.insert(self.X[0, :], 0, 0)[:-1, np.newaxis], self.X), axis=1)
                self.X = np.concatenate((self.X, np.append(self.X[:, -1], 0)[np.newaxis, :]), axis=0)
            else:
                self.X = np.concatenate((np.zeros((self.X.shape[1], 1)), self.X), axis=1)
                self.X = np.concatenate((self.X, np.zeros((1, self.X.shape[1]))), axis=0)

        # Do (0, 1] min-max normalization
        if normalize:
            minn = np.min(self.X)
            maxx = np.max(self.X)
            self.X = (self.X - minn) / (maxx - minn)

        self.X = torch.tensor(self.X, dtype=torch.float, device=device)

        if maxseqlen > len(self.X) or maxind >= len(self.X) or maxseqlen > (maxind + 1):
            raise Exception(f"maxseqlen and maxind must be < len(X): {maxseqlen} {maxind}")

        if padval >= 0 and padval <= 1:
            raise Exception(f"padval must not fall within normalized values [0, 1]: {padval}")

        # One past the used indices is used as the padding idx in the embeddings
        self.padind = maxind + 1
        self.padval = padval
        self.maxseqlen = maxseqlen

        # Fixed seed for reproducability
        rand = np.random.default_rng(seed=42)
        nsets = int(dsize / (maxseqlen - 1)) # number of sets with len [n, n-1, ..., 2]
        # The number of times to repeat the (shuffled) inds array to get all needed vals for the sets
        # Add 1 to compensate for rounding down
        nrepeat = np.ceil(sum(range(2, maxseqlen + 1)) * (nsets + 1) / (maxind + 1) + 1).astype(int)
        print(f"All (shuffled) inds are repeated {nrepeat} times to get full dataset.")
        # A sequence of all inds, shuffled each time, repeated for nrepeat
        # In order to ensure each ind shows up equal amount in total dataset
        inds = list(range(maxind + 1))
        allvals = np.concatenate([rand.permuted(inds) for i in range(nrepeat)])
        # Split allvals into sets of size 2, 3, ..., maxseqlen
        splitinds = np.cumsum(np.concatenate([range(2, maxseqlen + 1) for i in range(nsets + 1)]))

        self.mats = [torch.tensor(x, dtype=torch.int32, device=device)
                     for x in np.split(allvals, splitinds) if len(x) <= maxseqlen and len(x) > 1][:dsize]

        self._len = len(self.mats)
        assert self._len == dsize


    def __getitem__(self, idx):
        inds = self.mats[idx]
        vals = self.X[inds, :][:, inds]
        # Pad in embed/vocab dim the vals out to maxseqlen
        vals = F.pad(vals, (0, self.maxseqlen - len(inds)), value=self.padval)
        return (inds, vals)



class TokenizedCollateFn:
    """Needs to be a class so it can be pickled so that the stupid multithread DataLoader can work...
    """
    def __init__(self, padind: int, padval):
        self.padind = padind
        self.padval = padval

    def collate_fn(self, batch):
        ind_vecs, val_mats = tuple(zip(*batch))
        lens = [len(x) for x in ind_vecs]
        ind_vecs = pad_sequence(ind_vecs, batch_first=True, padding_value=self.padind)
        val_mats = pad_sequence(val_mats, batch_first=True, padding_value=self.padval)

        # In pytorch transformers, if a BoolTensor is provided, the positions with
        # the value of True will be ignored while the position with the value of False will be unchanged
        mask = torch.full(ind_vecs.shape, True).to(device)
        for i, l in enumerate(lens):
            mask[i, :l] = False

        return ind_vecs, val_mats, mask



class TokenizedPopDataset(TokenizedKingDataset):
    def __init__(self,
                 filepath: str,
                 labels: pd.Series,
                 index_col=False,
                 dsize=10000,
                 maxseqlen=2502,
                 maxind=2501):

        # Doesn't matter
        filled = True
        normalize = True

        if labels.dtype.kind != "i":
            raise Exception(f"Must pass in ints for y: {labels}")

        padval = labels.max() + 1

        super().__init__(filepath, index_col, filled, normalize, padval, dsize, maxseqlen, maxind, remove_starts=False)

        if len(labels) != len(self.X) or len(labels) < maxind:
            raise Exception(f"Datasets must match len(y) = {len(labels)}")

        # self.y = torch.tensor(pd.get_dummies(labels, dtype=int).to_numpy(), dtype=torch.int32, device=device)
        self.y = torch.tensor(labels.to_numpy(), dtype=torch.long, device=device)

    def __getitem__(self, idx):
        inds = self.mats[idx]
        vals = self.y[inds]
        return (inds, vals)




class RotatedMatrixDataset(FilledMatrixDataset):
    """Makes an output of a square (if no pad) matrix where, given idx and row of values [n...k] at idx, make
        len-pad x len matrix where all zeros except column idx has the values [n..k].

    Args:
        filepath (str): Where the king matrix is located, ends in .csv
        index_col (bool, optional): If there's an index column in the csv. Defaults to False.
        pad (int, optional): Add zero columns to embedding (aka col) dim. Defaults to 0.
        filled (bool, optional): Whether to use filled matrix. Defaults to False.
    """
    def __init__(self,
                 filepath: str,
                 index_col=False,
                 pad=0,
                 filled=False,
                 normalize=False):

        # long_dim_first doesn't matter since making square matrix

        # Get filled matrix
        if filled:
            super().__init__(filepath, index_col, pad, long_dim_first=False)
        # No filled matrix, just normal
        else:
            super(FilledMatrixDataset, self).__init__(filepath, no_zero=False, pad=pad, long_dim_first=False)

        # self.mats = []
        # for idx in range(self._len):
        #     vec = torch.tensor(self.X.iloc[idx, :],
        #                     dtype=torch.float,
        #                     device=device)

        #     # Make rotated matrix
        #     mat = torch.zeros((self._len, self._len + pad), dtype=torch.float, device=device)
        #     mat[:, idx] = vec[pad:]
        #     self.mats.append(mat.to_sparse())

        self.pad = pad

        # Convert to numpy array actually
        self.X = self.X.to_numpy()

        # Do (0, 1] min-max normalization
        if normalize:
            minn = np.min(self.X)
            maxx = np.max(self.X)
            self.X = (self.X - minn) / (maxx - minn)



    def __getitem__(self, idx):
        # return self.mats[idx]
        vec = torch.tensor(self.X[idx, :],
                            dtype=torch.float,
                            device=device)

        # Make rotated matrix
        mat = torch.zeros((self._len, self._len + self.pad), dtype=torch.float, device=device)
        mat[:, idx] = vec[self.pad:]
        return mat


class SimpleEmbedMatrixDataset(FilledMatrixDataset):
    """Make output where for given row n in king matrix len m, rotate row n to be col, and add a second
    zeros col where at ind n = 1. Final matrix m x 2

    Args:
        filepath (str): Where the king matrix is located, ends in .csv
        index_col (bool, optional): If there's an index column in the csv. Defaults to False.
        filled (bool, optional): Whether to use filled matrix. Defaults to False.
        normalize (bool, optional): Whether to normalize all to between 0-1. Defaults to False.
    """
    def __init__(self,
                 filepath: str,
                 index_col=False,
                 filled=False,
                 normalize=False,
                 mask: Callable = None):
        # long_dim_first doesn't matter since making 2 col matrix
        long_dim_first = False
        # No padding since only ever 2 cols
        pad = 0



        # Get filled matrix
        if filled:
            super().__init__(filepath, index_col, pad, long_dim_first=long_dim_first, mask=mask)
        # No filled matrix, just normal
        else:
            super(FilledMatrixDataset, self).__init__(filepath, no_zero=False, pad=pad, long_dim_first=long_dim_first)

        self.mask = mask

        # Convert to numpy array actually
        self.X = self.X.to_numpy()

        # Do (0, 1] min-max normalization
        if normalize:
            minn = np.min(self.X)
            maxx = np.max(self.X)
            self.X = (self.X - minn) / (maxx - minn)

        # Store new matrices
        self.mats = []
        for idx in range(self._len):
            vec = torch.tensor(self.X[idx, :],
                            dtype=torch.float,
                            device=device)

            # Make embed matrix
            mat = torch.zeros((self._len, 2), dtype=torch.float, device=device)
            mat[:, 0] = vec
            mat[idx, 1] = 1.0
            self.mats.append(mat)


    def __getitem__(self, idx):
        return self.apply_mask(idx, self.mats[idx])





def sparse_collate_fn(batch, *restbatch):
    if restbatch and len(restbatch) > 0:
        return torch.stack((batch, *restbatch)).to_dense()
    else:
        return torch.stack(batch).to_dense()


def mask_neg1(matlen, idx):
    return np.random.randint(matlen, size=20), -1
