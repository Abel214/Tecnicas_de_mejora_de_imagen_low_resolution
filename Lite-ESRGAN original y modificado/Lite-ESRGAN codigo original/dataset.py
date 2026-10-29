import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import (
    read_cv2,
    random_crop_cv2,
    random_blur_cv2,
    random_resize,
    random_noise_cv2,
    jpeg_compression_cv2,
    random_sinc_with,
)


class ImageDataset(Dataset):
    def __init__(self, data_path, crop_size, lr_size, sinc_prob=0.3, sinc_omega=(3.1415926 / 1.5, 3.1415926), sinc_kernel=(5, 15)):
        self.data_path = data_path
        self.image_names = glob.glob(os.path.join(data_path, '*'))
        self.num_samples = len(self.image_names)
        self.crop_size = crop_size
        self.lr_size = lr_size
        self.sinc_prob = sinc_prob
        self.sinc_omega = sinc_omega
        self.sinc_kernel = sinc_kernel

    def __getitem__(self, item):
        image = read_cv2(self.image_names[item])
        image = random_crop_cv2(image, self.crop_size)
        lr_image = random_blur_cv2(image)
        lr_image = random_resize(lr_image, self.lr_size)
        lr_image = random_noise_cv2(lr_image)
        lr_image = jpeg_compression_cv2(lr_image, (20, 100))
        lr_image = random_sinc_with(lr_image, self.sinc_prob, self.sinc_omega, self.sinc_kernel)
        return transforms.ToTensor()(lr_image), transforms.ToTensor()(image)

    def __len__(self):
        return self.num_samples


