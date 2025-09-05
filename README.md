# PreDDG

## 环境准备

```bash
  conda create -n PreDDG python=3.12
  conda activate PreDDG
  pip install numpy pandas scipy scikit-learn pathlib tqdm
  pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
  pip install tensorboard tensorboardX pytorch_lightning 
  pip install torch_geometric fair-esm
  pip install biopython MDAnalysis mdtraj 
```

## 数据准备

### 数据集下载

cDNA, PTMul数据集可以从[这里](https://github.com/jozhang97/MutateEverything)下载。DMS数据集可以从[Zenodo (Dieckhaus & Kuhlman, 2024)](https://zenodo.org/records/13345274)下载。

### 数据组织形式

数据集的文件夹结构如下：

```
data/
    dataset/
        ptmul/
            mutations/
                ptmul.csv
```

### 数据集预处理

#### 第一步：获取野生型蛋白质的PDB文件

下载各数据集中的野生型PDB文件，可以使用以下命令：

```bash
cd PreDDG

```

#### 第二步：获取PSSM

##### 准备blast及uniref90数据库

```bash
cd PreDDG/data/
conda install -c bioconda blast
wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
gzip -d uniref90.fasta.gz
makeblastdb -in uniref90.fasta -parse_seqids -hash_index -dbtype prot 
```

##### 进行序列比对

如果想要对指定的fasta文件进行比对，并且将比对结果保存到指定的文件夹中，可以使用下面的命令：

```bash
cd PreDDG
```

如果只是对本文的数据集进行序列比对，可以使用下面的命令：

```bash
cd PreDDG

```

#### 第三步：从ISM中提取特征

##### 准备ISM

ISM的官方仓库可以从[这里](https://github.com/jozhang97/ism)进入，选择下载[ISM-650M-UC30PDB](https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb)，并将其放置在`./data/ism/ism_t33_650M_uc30pdb/`目录下。

##### 采用ISM提取特征

```bash

```

## 训练模型

```bash
cd PreDDG
python main.py
```
