import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from facades_dataset import FacadesDataset
from GAN_network import UNetGenerator, PatchGANDiscriminator
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, criterion_adv, criterion_rec, device, epoch, num_epochs, epoch_g_train_losses, epoch_d_train_losses):
    """
    Train the GAN for one epoch.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        dataloader (DataLoader): DataLoader for the training data.
        g_optimizer (Optimizer): Optimizer for the generator.
        d_optimizer (Optimizer): Optimizer for the discriminator.
        criterion_adv (Loss): Adversarial loss function.
        criterion_rec (Loss): Reconstruction loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    generator.train()
    discriminator.train()

    running_g_loss = 0.0
    running_d_loss = 0.0
    lambda_rec = 0.4

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Generate fake images
        fake_semantic = generator(image_rgb)

        # ===================== Train the discriminator =====================
        # Zero the gradients
        d_optimizer.zero_grad()

        # Discriminator on real pairs
        real_pred = discriminator(image_rgb, image_semantic)
        real_loss = criterion_adv(real_pred, torch.ones_like(real_pred, device=device))

        # Discriminator on fake pairs
        fake_pred = discriminator(image_rgb, fake_semantic.detach())
        fake_loss = criterion_adv(fake_pred, torch.zeros_like(fake_pred, device=device))

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        # ===================== Train the generator =====================
        g_optimizer.zero_grad()
        adv_loss = criterion_adv(discriminator(image_rgb, fake_semantic), torch.ones_like(fake_pred, device=device))
        rec_loss = criterion_rec(fake_semantic, image_semantic)

        g_loss = (1-lambda_rec) * adv_loss + lambda_rec * rec_loss
        g_loss.backward()
        g_optimizer.step()

        # Update running loss
        running_d_loss += d_loss.item()
        running_g_loss += g_loss.item()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], '
              f'Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}')

        # Save sample outputs every 50 epochs
        if (epoch + 1) % 50 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_semantic, 'train_results', epoch + 1)

    # Print average losses
    avg_g_loss = running_g_loss / len(dataloader)
    avg_d_loss = running_d_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Discriminator Loss: {avg_d_loss:.4f}, Average Generator Loss: {avg_g_loss:.4f}')
    epoch_g_train_losses.append(avg_g_loss)
    epoch_d_train_losses.append(avg_d_loss)


def validate(generator, dataloader, criterion_rec, device, epoch, num_epochs, epoch_val_losses):
    """
    Validate the generator on the validation dataset.

    Args:
        generator (nn.Module): The generator model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion_rec (Loss): Reconstruction loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    generator.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Generate fake images
            fake_semantic = generator(image_rgb)

            # Compute reconstruction loss
            rec_loss = criterion_rec(fake_semantic, image_semantic)
            running_val_loss += rec_loss.item()

            # Save sample outputs every 5 epochs
            if (epoch + 1) % 25 == 0 and i == 0:
                save_images(image_rgb, image_semantic, fake_semantic, 'val_results', epoch + 1)

    # Calculate average validation loss
    avg_val_loss = running_val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
    epoch_val_losses.append(avg_val_loss)
    return avg_val_loss


def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=4)

    # Define the generator and discriminator models
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # Initialize model, loss function, and optimizer
    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_rec = nn.L1Loss()

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    g_scheduler = StepLR(g_optimizer, step_size=50, gamma=0.5)
    d_scheduler = StepLR(d_optimizer, step_size=50, gamma=0.5)

    # Training loop
    num_epochs = 500
    epoch_g_train_losses = []
    epoch_d_train_losses = []
    epoch_val_losses = []

    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion_adv, criterion_rec, device, epoch, num_epochs, epoch_g_train_losses, epoch_d_train_losses)
        validate(generator, val_loader, criterion_rec, device, epoch, num_epochs, epoch_val_losses)

        # 学习率更新
        g_scheduler.step()
        d_scheduler.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 50 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/GAN_g_model_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/GAN_d_model_epoch_{epoch + 1}.pth')

    # 绘制学习曲线
    plt.figure(figsize=(10, 5), dpi=300)
    # plt.plot(range(1, num_epochs + 1), epoch_g_train_losses, label='generator train Loss')
    plt.plot(range(1, num_epochs + 1), epoch_d_train_losses, label='discriminator train Loss')
    plt.plot(range(1, num_epochs + 1), epoch_val_losses, label='validate Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.title('Loss per Epoch')
    plt.legend()
    plt.savefig('GAN_cityscapes_epoch_loss.png')

if __name__ == '__main__':
    main()
