import os
import platform
import random
import time
import multiprocessing as mp
import numpy as np
import cv2
import torch
from torchvision import transforms


# Filesystem helpers
def insert_filename_ending(filename, ending):
    path, suffix = os.path.splitext(filename)
    return path + ending + suffix


# Image I/O and conversions
def read_cv2(filename, bgr_to_rgb=True):
    image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
    if bgr_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_cv2(filename, image, rgb_to_bgr=True):
    if rgb_to_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imencode(os.path.splitext(filename)[1], image)[1].tofile(filename)


def standard_image(image: np.ndarray):
    if image.dtype != np.uint8:
        return np.clip(np.round(image), 0, 255).astype(np.uint8)
    return image


def convolution(image, kernel, out_float=False):
    new_image = cv2.filter2D(image, -1, kernel)
    return new_image if out_float else standard_image(new_image)


# Blur/downsample/noise operations
def blur_cv2(image, radius):
    h, w, c = image.shape
    if isinstance(radius, tuple):
        radius = random.randint(max(radius[0], 1), radius[1])
    out = cv2.blur(image, (radius, radius))
    return out if c != 1 else out.reshape(h, w, 1)


def gaussian_blur_cv2(image, radius, sigma):
    h, w, c = image.shape
    if isinstance(radius, tuple):
        radius = random.randrange((max(radius[0], 1) // 2) * 2 + 1, radius[1] + 1, 2)
    if isinstance(sigma, tuple):
        sigma = random.uniform(sigma[0], sigma[1])
    out = cv2.GaussianBlur(image, (radius, radius), sigma)
    return out if c != 1 else out.reshape(h, w, 1)


def sinc_kernel_cv2(omega_c, N, normalize=True):
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = np.fromfunction(
            lambda x, y: omega_c
            * np.sinc(omega_c * np.sqrt((x - (N - 1) / 2) ** 2 + (y - (N - 1) / 2) ** 2) / np.pi)
            / (2 * np.pi)
            * omega_c,
            [N, N],
        )
    if N % 2:
        kernel[(N - 1) // 2, (N - 1) // 2] = omega_c ** 2 / (4 * np.pi)
    if normalize:
        kernel /= kernel.sum()
    return kernel


def sinc_cv2(image, omega_c, kernel_size, normalize=True, out_float=False):
    if isinstance(kernel_size, tuple):
        kernel_size = random.randrange((max(kernel_size[0], 1) // 2) * 2 + 1, kernel_size[1] + 1, 2)
    if isinstance(omega_c, tuple):
        omega_c = random.uniform(omega_c[0], omega_c[1])
    kernel = sinc_kernel_cv2(omega_c, kernel_size, normalize)
    return convolution(image, kernel, out_float)


def noise_cv2(image, range_val):
    h, w, c = image.shape
    image = image.astype(np.float64)
    if isinstance(range_val, tuple):
        range_val = random.uniform(range_val[0], range_val[1])
    noise = (np.random.rand(h, w, c) - 0.5) * range_val
    return standard_image(image + noise)


def gaussian_noise_cv2(image, sigma):
    h, w, c = image.shape
    image = image.astype(np.float64)
    if isinstance(sigma, tuple):
        sigma = random.uniform(sigma[0], sigma[1])
    noise = np.random.randn(h, w, c) * sigma
    return standard_image(image + noise)


def poisson_noise_cv2(image, mu, multiplier=1):
    h, w, c = image.shape
    image = image.astype(np.float64)
    if isinstance(mu, tuple):
        mu = random.uniform(mu[0], mu[1])
    if isinstance(multiplier, tuple):
        multiplier = random.uniform(multiplier[0], multiplier[1])
    noise = (np.random.poisson(mu, (h, w, c)) - mu) * multiplier
    return standard_image(image + noise)


def salt_noise_cv2(image, prob):
    h, w, c = image.shape
    image = image.astype(np.float64)
    if isinstance(prob, tuple):
        prob = random.uniform(prob[0], prob[1])
    noise = np.random.rand(h, w, c)
    image[noise < prob] = 255
    return standard_image(image)


def pepper_noise_cv2(image, prob):
    h, w, c = image.shape
    image = image.astype(np.float64)
    if isinstance(prob, tuple):
        prob = random.uniform(prob[0], prob[1])
    prob /= 2
    noise = np.random.rand(h, w, c)
    image[noise < prob] = 255
    image[noise > 1 - prob] = 0
    return standard_image(image)


def jpeg_compression_cv2(image, quality):
    if isinstance(quality, tuple):
        quality = random.randint(quality[0], quality[1])
    return cv2.imdecode(cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)


def random_crop_cv2(image, size):
    if isinstance(size, int):
        size = (size, size)
    h, w, _ = image.shape
    y, x = random.randint(0, h - size[1]), random.randint(0, w - size[0])
    return image[y:y + size[1], x:x + size[0], :]


def random_method_resize_cv2(image, size):
    interpolation = random.choice([
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
    ])
    return cv2.resize(image, size, interpolation=interpolation)


# Random degradations used by datasets
def random_blur_cv2(image):
    prob = 0.7
    x = random.random()
    if x <= prob / 4:
        return blur_cv2(image, (2, 6))
    elif x <= prob / 4 * 2:
        return gaussian_blur_cv2(image, (3, 15), (0.2, 3))
    elif x <= prob:
        return sinc_cv2(image, (np.pi / 3, np.pi), (5, 25), True)
    else:
        return image


def random_noise_cv2(image):
    prob = 0.9
    x = random.random()
    if x < prob / 5:
        return noise_cv2(image, (0, 120))
    elif x < prob / 5 * 2:
        return gaussian_noise_cv2(image, (0, 60))
    elif x < prob / 5 * 3:
        return poisson_noise_cv2(image, (0, 100), (0.5, 7))
    elif x < prob / 5 * 4:
        return salt_noise_cv2(image, (0, 0.035))
    elif x < prob:
        return pepper_noise_cv2(image, (0.0, 0.035))
    else:
        return image


def random_sinc_with(image, prob, omega_c_range, kernel_range):
    x = random.random()
    if x <= prob:
        return sinc_cv2(image, omega_c_range, kernel_range, True)
    return image


def random_resize(image, lr_size, prob=0.6):
    x = random.random()
    if x <= prob:
        image = random_method_resize_cv2(
            image,
            (random.randint(lr_size // 2, lr_size), random.randint(lr_size // 2, lr_size)),
        )
    return random_method_resize_cv2(image, (lr_size, lr_size))


# Tensor/image conversion
def image_to_tensor_cv2(image, device=None):
    if isinstance(image, str):
        image = read_cv2(image)
    output = transforms.ToTensor()(image).unsqueeze(0)
    return output.to(device) if device else output


def tensor_to_image_cv2(image, save_filename=None):
    output = (
        torch.clamp(torch.round(image * 255), 0, 255)
        .squeeze()
        .cpu()
        .detach()
        .numpy()
        .astype(np.uint8)
        .transpose((1, 2, 0))
    )
    if save_filename:
        write_cv2(save_filename, output)
    return output


def imagenet_transform_function():
    return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# Device and training utils
def training_device(cuda=True):
    # Basic platform info
    print(f'Operating system: {platform.platform()}')
    # CPU info via platform where available (best-effort)
    try:
        processor = platform.processor() or platform.machine()
        if processor:
            print(f'CPU: {processor}')
    except Exception:
        pass
    # CUDA availability
    if cuda:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print('Support CUDA')
            print(f'Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        else:
            print('Not support CUDA')
        print()
        return torch.device('cuda' if cuda_available else 'cpu')
    else:
        print()
        return torch.device('cpu')


def save_model(path, model, optimizer=None, epoch=None, test_loss=None, save_simplied_model=True):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    if isinstance(model, (list, tuple)):
        model_state_dict = [m.state_dict() for m in model]
    else:
        model_state_dict = [model.state_dict()]
    state = {'model': model_state_dict}
    if optimizer is not None:
        if isinstance(optimizer, (list, tuple)):
            optimizer_state_dict = [o.state_dict() for o in optimizer]
        else:
            optimizer_state_dict = [optimizer.state_dict()]
        state['optimizer'] = optimizer_state_dict
    if epoch is not None:
        state['epoch'] = epoch
    if test_loss is not None:
        if isinstance(test_loss, list):
            test_loss = min(test_loss)
        state['test_loss'] = test_loss
    torch.save(state, path)
    if save_simplied_model:
        state_simple = {'model': model_state_dict}
        name, suffix = os.path.splitext(path)
        torch.save(state_simple, f'{name}_{epoch}{suffix}')


def read_model(path, model, optimizer=None):
    epoch = 1
    test_loss = [100000]
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        if 'model' in checkpoint:
            if isinstance(model, (list, tuple)):
                if 'model2' in checkpoint:
                    model[0].load_state_dict(checkpoint['model'])
                    model[1].load_state_dict(checkpoint['model2'])
                else:
                    for i in range(len(checkpoint['model'])):
                        if model[i]:
                            model[i].load_state_dict(checkpoint['model'][i])
            else:
                if isinstance(checkpoint['model'], list):
                    model.load_state_dict(checkpoint['model'][0])
                else:
                    model.load_state_dict(checkpoint['model'])
        if (optimizer is not None) and ('optimizer' in checkpoint):
            if isinstance(optimizer, (list, tuple)):
                if 'optimizer2' in checkpoint:
                    optimizer[0].load_state_dict(checkpoint['optimizer'])
                    optimizer[1].load_state_dict(checkpoint['optimizer2'])
                else:
                    for i in range(len(checkpoint['optimizer'])):
                        optimizer[i].load_state_dict(checkpoint['optimizer'][i])
            else:
                if isinstance(checkpoint['optimizer'], list):
                    optimizer.load_state_dict(checkpoint['optimizer'][0])
                else:
                    optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch'] + 1
        if 'test_loss' in checkpoint:
            test_loss.append(checkpoint['test_loss'])
    return epoch, test_loss


# Prefetching dataloader
def _prefetch_put_data(dataloader, queue, buffer_size, wait_time):
    while True:
        for data in dataloader:
            while True:
                if queue.qsize() >= buffer_size:
                    time.sleep(wait_time)
                    continue
                else:
                    queue.put(data)
                    break


class PytorchDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, buffer_size=3, wait_time=0.05):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self.queue = mp.Queue()
        self.buffer_size = buffer_size
        self.wait_time = wait_time
        p = mp.Process(target=_prefetch_put_data, args=(self.dataloader, self.queue, self.buffer_size, self.wait_time))
        p.start()

    def get(self):
        return self.queue.get()

    def __len__(self):
        return len(self.dataloader)


