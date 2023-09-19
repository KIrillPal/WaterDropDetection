from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image, ImageFilter
from custom_transforms import ListRandomCrop
import numpy as np

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
    

def get_magnitude(image):
    # Kernels
    sobelGx_kernel =  np.asarray([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ]) * 1.
    sobelGy_kernel = sobelGx_kernel.T
    #%%
    # Apply Gx and Gy
    from scipy import signal
    Gx = signal.convolve2d(image, sobelGx_kernel)
    Gy = signal.convolve2d(image, sobelGy_kernel)
    # Calculate magnitude
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    magnitude = magnitude[1:-1, 1:-1]
    return magnitude.astype(np.float32)

class WaterDropDataset(Dataset):
    """Dataset with droplet masks on image lens"""

    def __init__(self, mode, image_dir, mask_dir, binarization=False, threshold=0.5, crop_shape=(64, 64)):
        """
        Arguments:
            mode (string): String with all using channels. 
                As example: "RGBS" means 4 channels: Red, Green, Blue and Saturation
                Channels:
                1) R, G, B - rgb channels
                2) H, S, V - hsv channels
                3) S - satuation channel
                4) D - dwt channel
                5) P - peak channel
                Their order in the output is set by their order in mode.
            image_dir (string): Directory with all the images.
            mask_dir (string):  Directory with all the masks.
            binarization (bool): Flag if binarization for mask is needed
            threshold (float): Threshold for binarization.
            crop_shape (tuple(h,w)): Shape to be cropped.
        """
        image_paths, mask_paths = load_stereo(image_dir, mask_dir)
        self.mode = mode
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.binarization = binarization
        self.threshold  = threshold
        self.crop_shape = crop_shape

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx]).convert('L')
        return self.transform(image, mask)
    
    def transform(self, image, mask, depth=4):
        crop = ListRandomCrop(self.crop_shape)
        to_tensor = transforms.ToTensor()
        image_norm = transforms.Normalize((0, 0, 0), (1, 1, 1))
        mask_norm = transforms.Normalize(0, 1)
        
        # Identical random crop for two images
        [image, mask] = crop([image, mask])
        # Computing additional layers for image
        if any(c in self.mode for c in 'HSV'):
            h, s, v = image.convert('HSV').split()
            if 'H' in self.mode:
                h = mask_norm(to_tensor(h))
            if 'S' in self.mode:
                s = mask_norm(to_tensor(s))
            if 'V' in self.mode:
                v = mask_norm(to_tensor(v))
        if 'D' in self.mode or 'I' in self.mode:
            dwt = to_tensor(self.__dwt_map(image, 3))
        if 'P' in self.mode:
            peak_np = self.__peak_map(image)
            peak = mask_norm(to_tensor(peak_np))

        image = image_norm(to_tensor(image))
        mask = mask_norm(to_tensor(mask))  
        
        # Combine channels
        channels = []
        for c in self.mode:
            match c:
                case 'R':
                    r = torch.stack([image[0]])
                    channels.append(r)
                case 'G':
                    g = torch.stack([image[1]])
                    channels.append(g)
                case 'B':
                    b = torch.stack([image[2]])
                    channels.append(b)
                case 'H':
                    channels.append(h)
                case 'S':
                    channels.append(s)
                case 'V':
                    channels.append(v)
                case 'D':
                    channels.append(dwt)
                case 'I':
                    channels.append(dwt)
                case 'P':
                    channels.append(peak)
        features = torch.cat(channels, 0)
        
        # Binarization
        if self.binarization:
            binarize = lambda x: x > self.threshold
            mask.apply_(binarize)
        return features, mask

    def random_split(self, val_percent=0.15):
        val_size = int(len(self) * val_percent)
        train_size = len(self) - val_size
        return torch.utils.data.random_split(self, [train_size, val_size])

    @staticmethod
    def __rgb_to_s(image):
        h, s, v = image.convert('HSV').split()
        return s

    @staticmethod
    def __dwt_map(image, level=2, brightness=5):
        import pywt
        image = np.asarray(image.convert('L'))
        dwt = pywt.wavedec2(image, 'haar', level=level)
        map = dwt[0] / brightness
        for LH, HL, HH in dwt[1:]:
            map = np.concatenate((map, LH), axis=0)
            col = np.concatenate((HL, HH), axis=0)
            map = np.concatenate((map, col), axis=1)
        if np.ptp(map) > 0: 
            map = (map - np.min(map))/np.ptp(map)
        return map.astype('float32')

    @staticmethod
    def __dwt_levels(image, level=2):
        import pywt
        image = np.asarray(image.convert('L'))
        dwt = pywt.wavedec2(image, 'haar', level=level)
        return [np.concatenate([LH, HL, HH]).astype('float32') 
                for LH, HL, HH in dwt[1:]
               ]

    @staticmethod
    def __ptp_map(image, window=(23, 23)):
        image = np.asarray(image.convert('L'))
        from scipy.ndimage.filters import maximum_filter, minimum_filter
        h,w = window
        H,W = image.shape
        # Use 2D max filter
        maxs = maximum_filter(image, size=window) 
        mins = minimum_filter(image, size=window)
        
        return maxs-mins
        
    @staticmethod
    def __peak_map(image, pr=3, br=22):
        blur = ImageFilter.BoxBlur
        peaks = np.asarray(image.filter(blur(pr)))
        backs = np.asarray(image.filter(blur(br)))
        map = np.mean(abs(peaks - backs), axis=2)
        if np.ptp(map) > 0: 
            map = (map - np.min(map))/np.ptp(map)
        return get_magnitude(map)
