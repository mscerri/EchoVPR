# EchoVPR

Repo for **EchoVPR: Echo State Networks for Visual Place Recognition**

Pre-print is available at https://arxiv.org/abs/2110.05572

**Currently under development**

Dirs: 
- `data`: pre-collected hidden representations
- `datasets`: raw RGB images
- `src`: includes dataset, utils, ESNs, and NetVLAD implementation 

## Setup
### PIP
```bash
pip install -r requirements.txt
```

## Run

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
python extract_hidden_repr.py --config_file=configs\train_mlp_nordland.ini --artifact_name={wandb_artifact_name} --model_name=model.pt
```
or on the Oxford dataset:
```python
python extract_hidden_repr.py --config_file=configs\train_mlp_oxford.ini --artifact_name={wandb_artifact_name} --model_name=model.pt
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

### Evaluate
Evaluate the performance of ESN with or without Patch-NetVLAD applied and reports the achieved Recall@N
```
python evaluate_esn.py \
    --config_file=configs\train_esn_oxford.ini \
    --patchnetvlad_config_file=configs\eval_patchnetvlad.ini \
    --artifact_name={wandb_artifact_name} \
    --test_input_features_dir={directory_containing_local_features_for_test} \
    --val_input_features_dir={directory_containing_local_features_for_val} \
    --index_input_features_dir={directory_containing_local_features_for_index}
```

In addition to the options described above all commands have the following optional arguments:
* `--project`: overrides the default WandB project
* `--entity`: overrides the default WandB entity

## Acknowledgement
The code for in this repository is based on [Nanne/pytorch-NetVlad](https://github.com/Nanne/pytorch-NetVlad) and Patch-NetVLAD on [QVPR/Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD).