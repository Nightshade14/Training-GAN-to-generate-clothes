import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 7*7*256, bias=False),
            nn.BatchNorm1d(7*7*256),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128*7*7, 1)
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizers
lr = 1e-4
opt_g = optim.Adam(generator.parameters(), lr=lr)
opt_d = optim.Adam(discriminator.parameters(), lr=lr)



# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# Creating a GAN training routine
def train_gan(epochs, dataloader):
    fixed_noise = torch.randn(64, 100, device=device)
    generator_losses = []
    discriminator_losses = []

    for epoch in range(epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.size(0)

            # Labels for real and fake images
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # Discriminator training
            discriminator.zero_grad()
            outputs_real = discriminator(images).squeeze()
            loss_real = criterion(outputs_real, real_labels)

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)
            outputs_fake = discriminator(fake_images.detach()).squeeze()
            loss_fake = criterion(outputs_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # Generator training
            generator.zero_grad()
            outputs_fake = discriminator(fake_images).squeeze()
            loss_g = criterion(outputs_fake, real_labels)
            loss_g.backward()
            opt_g.step()

            # Track losses
            g_loss_epoch += loss_g.item()
            d_loss_epoch += loss_d.item()

        generator_losses.append(g_loss_epoch / len(dataloader))
        discriminator_losses.append(d_loss_epoch / len(dataloader))

        print(f"Epoch [{epoch+1}/{epochs}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

        if epoch + 1 in [10, 30, 50]:
            with torch.no_grad():
                fake_samples = generator(fixed_noise).cpu()
                grid = make_grid(fake_samples, nrow=8, normalize=True)
                plt.imshow(grid.permute(1, 2, 0))
                plt.title(f"Epoch {epoch+1}")
                plt.show()

    return generator_losses, discriminator_losses
    
# Train the GAN for 50 epochs
generator_losses, discriminator_losses = train_gan(50, dataloader)