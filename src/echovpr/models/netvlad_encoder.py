import torch.nn as nn
from echovpr.trainer.utils.netvlad_utils import \
    load_netvlad_model_from_checkpoint


class NetVLADEncorder(nn.Module):
    """NetVLAD Network"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pool_size = int(config['model_num_pcs'])
        self.model = load_netvlad_model_from_checkpoint(config)

    def forward(self, x):
        image_encoding = self.model.encoder(x)

        if self.config['model_pooling'].lower() == 'patchnetvlad':
            vlad_local_encoding, vlad_global_encoding = self.model.pool(image_encoding)

            global_pca_encoding = self.model.WPCA(vlad_global_encoding.unsqueeze(-1).unsqueeze(-1))

            local_pca_encodings = []
            for this_iter, this_local in enumerate(vlad_local_encoding):
                this_patch_size = self.model.pool.patch_sizes[this_iter]

                this_local_pca = self.model.WPCA(this_local.permute(2, 0, 1).reshape(-1, this_local.size(1)).unsqueeze(-1).unsqueeze(-1)).\
                        reshape(this_local.size(2), this_local.size(0), self.pool_size ).permute(1, 2, 0)

                local_pca_encodings.append((this_patch_size, this_local_pca))
            
            return global_pca_encoding, local_pca_encodings
        else:
            vlad_encoding = self.model.pool(image_encoding)
            pca_encoding = self.model.WPCA(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
            return pca_encoding
