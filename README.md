# PreDDG

## 📦 Environment Setup
We recommend creating a dedicated [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:

```bash
  conda create -n PreDDG python=3.12
  conda activate PreDDG
  pip install numpy pandas scipy scikit-learn pathlib tqdm
  pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
  pip install tensorboard tensorboardX pytorch_lightning 
  pip install torch_geometric fair-esm
  pip install biopython
```
⚠️ Please ensure that the CUDA version matches your PyTorch and torch-geometric installation. For details, refer to the [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)。

## 📂 Data Preparation

### Datasets


| Dataset  | Download Link                                                                               |
|----------|--------------------------------------------------------------------------------------|
| cDNA     | https://github.com/jozhang97/MutateEverything/                                       |
| cDNA2    | https://github.com/jozhang97/MutateEverything/                                       |
| PTMul-NR | https://ddgemb.biocomp.unibo.it/datasets/                                            |
| M28      | https://github.com/GenScript-IBDPE/UniMutStab/tree/main/Dataset/Independent/multiple |
| M38      | https://github.com/GenScript-IBDPE/UniMutStab/tree/main/Dataset/Independent/multiple |

Datasets should be placed under `./data/dataset/` directory. The folder structure should be as follows:

```
data/
    dataset/
        M28/
            mutations/
                M28.csv
```


### ISM Model Preparation

Download [ISM-650M-UC30PDB](https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb)，and place it in `./data/ism/ism_t33_650M_uc30pdb/` directory:
```
data/
    ism/
        ism_t33_650M_uc30pdb/
            config.json
            gitattributes
            ism_t33_650M_uc30pdb.pth
            model.safetensors
            special_tokens_map.json
            tokenizer_config.json
            vocab.txt
```

## 🚀 Running PreDDG for Prediction
Example: predicting on M28 dataset. Input files should be in .csv format with one of the following formats:

**Format 1** (with both wild-type and mutant sequences):

| pdb_id | wt_seq | mut_info | mut_seq |
|--------|--------|--------|--------|


**Format 2** (only mutation info provided):

| pdb_id | wt_seq | mut_info |
|--------|--------|--------|


**Note:**
 1. `mut_info` follows the format `WT_POS_MUT`, e.g., `Y68R` means the 68th position changes from Y to R.
 2. Multiple mutations are separated by `:`, e.g., `Y68R:A120V`.
 3. `mut_seq` is optional. If not provided, it will be computed based on `wt_seq` and `mut_info`.

```bash
cd PreDDG
python predict.py --test_name='M28' --device='cuda:0'
```
Predictions are saved under `./data/dataset/M28/predictions/`. Example output:

| pdb_id | wt_seq | mut_info | mut_seq | preddg |
|--------|--------|--------|--------|--------|

For more details, please refer to the paper and source code.

## 📖 Citation
If you find PreDDG useful, please cite our paper:
```bibtex
@article{
  title={},
  author={},
  journal={},
  year={}
}
```
