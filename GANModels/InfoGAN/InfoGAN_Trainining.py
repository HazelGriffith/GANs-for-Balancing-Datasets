"""
@File: Info GAN Training on DeepInsight Dataset
@Author: Puxin
@Last Modified: 2024-12-03

Base Code:
    - This code is based on [Erik Linder-Norén/PyTorch-GAN/implementations/infogan/infogan.py]
      (https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan/infogan.py)
    - Modifications made by Puxin to adapt the code for training on the DeepInsight dataset
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import numpy as np
import itertools

import torchvision.transforms as transforms 

import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

set_seed(2024)

# InfoGAN Definations
channels = 1 # number of image channels
num_classes = 1 # number of classes for dataset
code_dim = 2 # latent code
latent_dim = 62 # dimensionality of the latent space
img_size = 32 # size of each image dimension

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = latent_dim + num_classes + code_dim

        self.init_size = img_size // 4  
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, num_classes), nn.Softmax(dim=1)) 
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

from torch.autograd import Variable

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))

# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)
categorical_loss.to(device)
continuous_loss.to(device)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Load the dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def convert_I_mode(image):
    # Convert 'I' mode images to 'L'
    if image.mode == 'I':
        image = image.convert('L')
    return image

# Define transformations with channel = 1 and normalization
transform = transforms.Compose([
    transforms.Lambda(convert_I_mode),            
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),                        
    transforms.Normalize([0.5], [0.5])])

dataset_path = "/deepinsight-gan/infogan"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

lr = 0.0002 # adam: learning rate
b1 = 0.5 # adam: decay of first order momentum of gradient
b2 = 0.999 # adam: decay of first order momentum of gradient

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=lr, betas=(b1, b2)
)


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

# FloatTensor = torch.cuda.FloatTensor if mps else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if mps else torch.LongTensor

# Training

n_epochs = 100
dataloader = data_loader
sample_interval = 400 
avg_d_loss=[]
avg_g_loss=[]
avg_info_loss=[]

torch.manual_seed(2024)
np.random.seed(2024)

for epoch in range(n_epochs):
    d_losses=[]
    g_losses=[]
    info_losses=[]
    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device) 
    
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device) 
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device) 

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor)).to(device) 
        labels = to_categorical(labels.cpu().numpy(), num_columns=num_classes) 

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device) 
        label_input = to_categorical(np.random.randint(0, num_classes, batch_size), num_columns=num_classes).to(device) 
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, code_dim)))).to(device) 

        # Generate a batch of images
        gen_imgs = generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Information Loss
        # ------------------

        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = np.random.randint(0, num_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False).to(device) 

        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))).to(device) 
        label_input = to_categorical(sampled_labels, num_columns=num_classes).to(device) 
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, code_dim)))).to(device)

        gen_imgs = generator(z, label_input, code_input) 
        _, pred_label, pred_code = discriminator(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input
        )

        info_loss.backward()
        optimizer_info.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        info_losses.append(info_loss.item())
    avg_d_loss.append(np.mean(d_losses))
    avg_g_loss.append(np.mean(g_losses))
    avg_info_loss.append(np.mean(info_losses))
    
    # --------------
    # Log Progress
    # --------------
    
    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
        % (epoch+1, n_epochs, np.mean(d_losses), np.mean(g_losses), np.mean(info_losses))
    )
    torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'epoch': epoch
    }, f"infogan_epoch{epoch+1}.pth")
    torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'epoch': epoch
    }, f"infogan_epoch{epoch+1}.pt")

import matplotlib.pyplot as plt

# Plotting the loss curve
plt.figure(figsize=(30, 5))
plt.subplot(1, 3, 1)
plt.plot(range(1, n_epochs + 1), avg_d_loss, label='D Loss', color='blue')
plt.plot(range(1, n_epochs + 1), avg_g_loss, label='G Loss', color='orange')
plt.plot(range(1, n_epochs + 1), avg_info_loss, label='Info Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
