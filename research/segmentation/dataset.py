from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from custom_transforms import ListRandomCrop

def load_stereo(image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_paths = []
    mask_paths  = []

    for path in image_dir.glob('*.png'):
        # check if not clear
        if path.name[-9:-6] != '000':
            mask_path = mask_dir / path.name
            image_paths.append(path)
            mask_paths.append(mask_path)
    return image_paths, mask_paths


class WaterDropDataset(Dataset):
    """Dataset with droplet masks on image lens"""

    def __init__(self, image_dir, mask_dir, crop_shape=(64, 64)):
        """
        Arguments:
            image_dir (string): Directory with all the images.
            mask_dir (string):  Directory with all the masks.
            crop_shape (tuple(h,w)): Shape to be cropped.
        """
        image_paths, mask_paths = load_stereo(image_dir, mask_dir)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.crop_shape = crop_shape

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx]).convert('L')
        return self.transform(image, mask)

    def transform(self, image, mask):
        crop = ListRandomCrop(self.crop_shape)
        to_tensor = transforms.ToTensor()
        image_norm = transforms.Normalize((0, 0, 0), (1, 1, 1))
        mask_norm = transforms.Normalize(0, 1)

        [image, mask] = crop([image, mask])
        image = image_norm(to_tensor(image))
        mask = mask_norm(to_tensor(mask))
        return image, mask

    def random_split(self, val_percent=0.15):
        val_size = int(len(self) * val_percent)
        train_size = len(self) - val_size
        return torch.utils.data.random_split(self, [train_size, val_size])