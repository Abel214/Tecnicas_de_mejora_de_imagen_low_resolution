import torch
import torch.nn as nn
import os

from utils import *
from dataset import ImageDataset
from model import SRNet, VGGNet, Discriminator

num_step = 100000
batch_size = 1
crop_size = 512
lr_size = crop_size // 4
num_step_update = 4
num_step_test = 400
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
pre_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model_base')
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model_gan')
test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_gan')


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
    dataset = ImageDataset(data_path, crop_size, lr_size, sinc_prob=0.4, sinc_omega=(np.pi / 2.2, np.pi), sinc_kernel=(5, 15))
    dataloader = PytorchDataLoader(dataset, batch_size)

    # Build models
    model = SRNet().to(device).train()
    vgg = VGGNet().to(device).eval()
    D = Discriminator().to(device).train()

    # Define optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.00005)
    optimizer.zero_grad()

    # Load checkpoints
    step, test_loss = read_model(os.path.join(model_path, 'model.pt'), [model, D], [optimizer, optimizer_D])

    while step <= num_step:
        # Get a batch
        lr_image, target_image = dataloader.get()
        # Move to device
        n = lr_image.shape[0]
        lr_image = lr_image.to(device)
        target_image = target_image.to(device)
        # Forward pass
        pred_image = model(lr_image)
        # Discriminator loss
        real_score = D(target_image.clone().detach()).view(n, -1).mean(1)
        real_label = torch.ones_like(real_score, dtype=torch.float32).to(device)
        loss_real = l2_loss(real_score, real_label)
        fake_score = D(pred_image.clone().detach()).view(n, -1).mean(1)
        fake_label = torch.zeros_like(fake_score, dtype=torch.float32).to(device)
        loss_fake = l2_loss(fake_score, fake_label)
        loss_d = loss_real + loss_fake
        # Optimize discriminator
        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()

        # Pixel loss
        loss_pixel = l1_loss(pred_image, target_image)
        # Perceptual loss
        pred_feature = vgg(transform_imagenet(pred_image))
        target_feature = vgg(transform_imagenet(target_image))
        loss_perceptual = 0
        for f1, f2 in zip(pred_feature, target_feature):
            loss_perceptual += l2_loss(f1, f2)
        loss_perceptual /= 100
        # GAN loss
        fake_score = D(pred_image).view(n, -1).mean(1)
        real_label = torch.ones_like(fake_score, dtype=torch.float32).to(device)
        loss_gan = l2_loss(fake_score, real_label) / 100
        # Total loss
        loss = loss_pixel + loss_perceptual + loss_gan
        (loss / num_step_update).backward()
        # Optimize generator
        if step % num_step_update == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Log info
        print(f'step: {step}/{num_step}  loss: {loss:.6f}  loss_pixel: {loss_pixel:.6f}  '
              f'loss_perceptual: {loss_perceptual:.6f}  loss_gan: {loss_gan:.6f}  '
              f'loss_real: {loss_real:.6f}  loss_fake: {loss_fake:.6f}')

        if step % num_step_test == 0:
            # Save checkpoint
            save_model(os.path.join(model_path, 'model.pt'), [model, D], [optimizer, optimizer_D], step)

            # Test image
            model.eval()

            image = read_cv2(os.path.join(test_data_path, 'sample.png'))
            image = image_to_tensor_cv2(image, device)
            with torch.no_grad():
                output = model(image)
            output = tensor_to_image_cv2(output, os.path.join(test_path, f'train_{step}.png'))

            model.train()

        step += 1
