from os.path import join

from echovpr.datasets.utils import input_transform
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, images_names, dataset_root_dir, config):
        super().__init__()

        self.images = [join(dataset_root_dir, image) for image in images_names]

        print('Number of images loaded: %d' % len(self.images))

        self.resize = (int(config['dataset_imageresizeh']), int(config['dataset_imageresizew']))
        self.transform = input_transform(self.resize)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)
