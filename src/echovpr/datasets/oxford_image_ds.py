from os.path import join

from echovpr.datasets.utils import input_transform
from PIL import Image
from robotcarsdk.camera_model import CameraModel
from robotcarsdk.image import load_image
from torch.utils.data import Dataset


class OxfordImageDataset(Dataset):
    def __init__(self, images_names, dataset_root_dir, config):
        super().__init__()

        self.images = [join(dataset_root_dir, image) for image in images_names]

         # pass in 1 imagename for the camera model to identify the correct camera used
        self.camera_model = CameraModel(join(dataset_root_dir, 'oxford', 'models'), images_names[0])
        
        print('Number of images loaded: %d' % len(self.images))

        self.resize = (int(config['dataset_imageresizeh']), int(config['dataset_imageresizew']))
        self.transform = input_transform(self.resize)

    def get_image(self, index):
        return load_image(self.images[index], model=self.camera_model)

    def get_transformed_image(self, index):
        img = self.get_image(index)
        img = img[:-160, :, :] # trim to remove car hood
        return self.transform(Image.fromarray(img))

    def __getitem__(self, index):
        return self.get_transformed_image(index)

    def __len__(self):
        return len(self.images)
