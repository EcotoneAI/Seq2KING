# Seq2KING

<p float="left" align="middle">
  <img src="Images/example_layer1.gif" width="300" />
  <img src="Images/example_layer2.gif" width="300" />
</p>

<p align="middle">

![ ](Images/Attention_bert_inverted.png)

</p>

<p align="middle">
Public code and data repository for the paper "Seq2KING: An unsupervised internal transformer representation of global human heritages"
</p>

---

## Hardware/Software Requirements

We used an A6000 GPU on cloud infrastructure (Azure, Paperspace) to train using the provided notebooks. You may suffice with lower ability GPU but we recommend the following:

- **OS:** Linux or macOS
- **GPU:** NVIDIA GPU with ≥16 GB VRAM (for training)
- **RAM:** ≥32 GB
- **Storage:** ≥50 GB free
- **Python:** 3.10+
- **Dependencies:**
  - Install via `pip install` the packages listed in the notebooks you want to run, and those used in `src/`.
  - Key packages: `torch`, `transformers`, `umap-learn`, `pandas`, `numpy`, `matplotlib`, `bertviz`
- **External tools used:**
  - KING kinship software ([Manichaikul et al. 2010](https://doi.org/10.1093/bioinformatics/btq559))

## Data

All raw genotype data are sourced from the [**1000 Genomes Project**](https://www.internationalgenome.org/) (GRCh37, Chr 1).

- **Processed kinship matrices** are stored in `Data/king_matrix_withnames.csv` and `Data/king_matrix.csv`; `king_matrix_withnames.csv` contains the ID codes of the human individuals for whom the row+column kinship value corresponds to.
- **Population and other info per individual** is in `Data/igsr_samples.csv`.

## Models

The model is a self-attention transformer-based architecture that learns to represent human genetic population info as embeddings, training by returning kinship inference.

### Training

To train the Seq2KING model used and analyzed in the paper, run `Notebooks/Train_16.ipynb`. It contains the code to: load the data, format it appropriately, create the model with the exact parameters and hyperparameters used in the paper, train it, and save the model checkpoints (at `Data/Output/Runs/`, which needs to be created prior to notebook running).

### Source Code

Most of the logic behind the above operations are abstracted into a custom python library. All python code pertaining to the Seq2KING model is in the `src/lib/` directory.

## Analysis/Inference/Comparison

The following notebooks have all the analyses referenced or discussed in the paper, as well as how to run trained models for inference:

- [KING_Data_Analysis](Notebooks/KING_Data_Analysis.ipynb): Demonstrates how to load the KING kinship matrix data, and performs various analyses on it, including population clustering and UMAP.
- [LGM_Analysis](Notebooks/LGM_Analysis.ipynb): Examples of loading models, along with initial attention analysis.
- [Tokenized_Analysis](Notebooks/Tokenized_Analysis.ipynb): Final model embedding analysis, attention analysis, and numerical analyses.

## Directory Structure

```
Seq2KING/
├── Data/ # KING matrix data
│
├── Notebooks/ # Training of models, and general analysis
│
├── src/
│ ├── lib/ # Custom python library for Seq2KING
│
├── Output/Runs/ # (Optional) store checkpoints here
│
├── Images/ # Images for this README
│
├── README.md # This file
```
