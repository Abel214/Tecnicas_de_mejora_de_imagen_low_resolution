import numpy as np
import torch
import torch.nn as nn

from utils import *
from dataset import ImageDataset
from model import SRNet, VGGNet

num_step = 40000
batch_size = 2
crop_size = 384
lr_size = crop_size // 4
num_step_update = 2
num_step_test = 400
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model_base')
test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_base')


if __name__ == '__main__':
    # Select device
    device = training_device()

    # Create directories
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # ImageNet normalization transform
    transform_imagenet = imagenet_transform_function()

    # Define loss functions
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    # Load dataset
    dataset = ImageDataset(data_path, crop_size, lr_size, sinc_prob=0.3, sinc_omega=(np.pi / 1.5, np.pi), sinc_kernel=(5, 15))
    dataloader = PytorchDataLoader(dataset, batch_size)

    # Build models
    model = SRNet().to(device).train()
    vgg = VGGNet().to(device).eval()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    optimizer.zero_grad()

    # Load checkpoints
    step, test_loss = read_model(os.path.join(model_path, 'model.pt'), model, optimizer)

    while step <= num_step:
        # Get a batch
        lr_image, target_image = dataloader.get()
        # Move to device
        n = lr_image.shape[0]
        lr_image = lr_image.to(device)
        target_image = target_image.to(device)
        # Forward pass
        pred_image = model(lr_image)
        # Pixel loss
        loss_pixel = l1_loss(pred_image, target_image)
        # Perceptual loss
        pred_feature = vgg(transform_imagenet(pred_image))
        target_feature = vgg(transform_imagenet(target_image))
        loss_perceptual = 0
        for f1, f2 in zip(pred_feature, target_feature):
            loss_perceptual += l2_loss(f1, f2)
        loss_perceptual /= 100
        # Total loss
        loss = loss_pixel + loss_perceptual
        (loss / num_step_update).backward()
        # Optimize model
        if step % num_step_update == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Log info
        print(f'step: {step}/{num_step}  loss: {loss:.6f}  loss_pixel: {loss_pixel:.6f}  loss_perceptual: {loss_perceptual:.6f}')

        if step % num_step_test == 0:
            # Save checkpoint
            save_model(os.path.join(model_path, 'model.pt'), model, optimizer, step)

            # Test image
            model.eval()

            image = read_cv2(os.path.join(test_data_path, 'sample.png'))
            image = image_to_tensor_cv2(image, device)
            with torch.no_grad():
                output = model(image)
            output = tensor_to_image_cv2(output, os.path.join(test_path, f'train_{step}.png'))

            model.train()

        step += 1
