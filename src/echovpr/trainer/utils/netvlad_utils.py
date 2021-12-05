from os.path import isfile, join

import torch
from configs import ROOT_DIR
from patchnetvlad.models.models_generic import get_backend, get_model


def load_netvlad_model_from_checkpoint(config):
    """
    Loads a pretrained netvlad model from a checkpoint.
    """

    resume_ckpt = config['model_resumepath'] + config['model_num_pcs'] + '.pth.tar'

    resume_ckpt = join(ROOT_DIR, resume_ckpt)
    print(f'Resume Checkpoint file {resume_ckpt}')
    assert isfile(resume_ckpt)

    print("=> loading checkpoint '{}'".format(resume_ckpt))
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['model_num_pcs'])
    config['model_num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

    encoder_dim, encoder = get_backend()
    netvlad_model = get_model(encoder, encoder_dim, config, append_pca_layer=True)
    netvlad_model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(resume_ckpt, ))

    for _, layer in netvlad_model.named_modules():
        for p in layer.parameters():
            p.requires_grad = False

    return netvlad_model
    