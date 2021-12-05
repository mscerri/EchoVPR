# EchoVPR

Repo for **EchoVPR: Echo State Networks for Visual Place Recognition**

Pre-print is available at https://arxiv.org/abs/2110.05572

## Quick Start

For a quick start first:
- Install all requirements `pip install -r requirements.txt`
- Download pre-trained models from [here](https://drive.google.com/file/d/1o9nuBByZ_XzbAUJ5C0JfzunFsoBw-VPC/view?usp=sharing) and extract them to `src\pretrained_models`
- Download data pre-processed representations from [here](https://drive.google.com/file/d/1TgLlNIUURswSkAsCWMUaFIkAHf3tRaU9/view?usp=sharing) and extract them to `src\data`

In the sections below a few example are given on how to reproduce the results.
 
## Evaluate

Evaluate the performance with or without Patch-NetVLAD applied and reports the achieved Recall@N

The commands below will reproduce the results in Table 1.

```
# Nordland Summer vs Winter - NV-ESN
python evaluate_esn.py --config_file=configs\eval_best_esn_nordland.ini --checkpoint_dir=pretrained_models\nordland_netvlad_esn

# Nordland Summer vs Winter - NV-SPARCE-ESN
python evaluate_esn.py --config_file=configs\eval_best_esn_sparce_nordland.ini --checkpoint_dir=pretrained_models\nordland_netvlad_esn_sparce

# Oxford, Day vs Night - NV-ESN
python evaluate_esn.py --config_file=configs\eval_best_esn_oxford.ini --checkpoint_dir=pretrained_models\oxford_netvlad_esn

# Oxford, Day vs Night - NV-SPARCE-ESN
python evaluate_esn.py --config_file=configs\eval_best_esn_sparce_oxford.ini --checkpoint_dir=pretrained_models\oxford_netvlad_esn_sparce
```

In addition to the above, if one would like to also obtain the results for `PatchL` denoted models, firstly the local features for both datasets would need to be generated. For more details on how to extract the local features using Patch-NetVLAD see [here](https://github.com/QVPR/Patch-NetVLAD#feature-extraction). 

An example of how to get the result for one model is given below.

```
# Nordland Summer vs Winter - PatchL-ESN
python evaluate_esn.py \
    --config_file=configs\eval_best_esn_nordland.ini \
    --checkpoint_dir=pretrained_models\nordland_netvlad_esn \
    --patchnetvlad_config_file=configs\eval_patchnetvlad.ini \
    --test_input_features_dir={directory_containing_local_features_for_test} \
    --val_input_features_dir={directory_containing_local_features_for_val} \
    --index_input_features_dir={directory_containing_local_features_for_index}
```

### Extracting NetVLAD descriptors
To extract NetVLAD descriptors required to preceed with the additional steps on the Nordland dataset:
```python
python export_netvlad_repr.py --config_file=configs\train_mlp_nordland.ini
```
or on the Oxford dataset:
```python
python export_netvlad_repr.py --config_file=configs\train_mlp_oxford.ini
```

### Training the Multi-Layer Perceptron
To train a multi-layer perceptron on the Nordland dataset:
```python
python train_mlp.py --config_file=configs\train_mlp_nordland.ini
```
or on the Oxford dataset:
```python
python train_mlp.py --config_file=configs\train_mlp_oxford.ini
```

### Extracting Hidden Layer descriptors
To extract Hidden Layer descriptors using a previously trained multi-layer perceptron on the Nordland dataset:
```python
python extract_hidden_repr.py --config_file=configs\train_mlp_nordland.ini --checkpoint_dir=pretrained_models\nordland_mlp
```
or on the Oxford dataset:
```python
python extract_hidden_repr.py --config_file=configs\train_mlp_oxford.ini --checkpoint_dir=pretrained_models\oxford_mlp
```

### Training the EchoStateNetwork
To train an EchoStateNetwork on the Nordland dataset:
```python
python train_esn.py --config_file=configs\train_esn_nordland.ini
```
or on the Oxford dataset:
```python
python train_esn.py --config_file=configs\train_esn_oxford.ini
```

## Acknowledgement
The code for NetVLAD in this repository is based on [Nanne/pytorch-NetVlad](https://github.com/Nanne/pytorch-NetVlad) and Patch-NetVLAD on [QVPR/Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD).