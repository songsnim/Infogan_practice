import torch 
import torch.nn as nn
from config import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
# dataloader

def load_MNIST():
    my_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5),   # 3 for RGB channels이나 실제론 gray scale
                                            std=(0.5))])  # 3 for RGB channels이나 실제론 gray scale
    train_data = datasets.MNIST(root='data/', train=True, transform=my_transform, download=True)
    test_data  = datasets.MNIST(root='data/', train=False, transform=my_transform, download=True)
    train_loader = data.DataLoader(train_data, BATCH_SIZE, shuffle=True)
    test_loader  = data.DataLoader(test_data, BATCH_SIZE, shuffle=True)
    return train_data, test_data, train_loader, test_loader

# Initializes weights according to the DCGAN paper
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)

def shape_test(enc, gen, critic,device):
    enc = enc.to(device)
    gen = gen.to(device)
    critic = critic.to(device)
    N, in_channels, H, W = BATCH_SIZE, 1, 28, 28
    x = torch.randn((N, in_channels, H, W)).to(device)
    assert critic(x).shape == (N, 1), "Discriminator test failed"
    enc = nn.Sequential(*(list(enc.children())[:5])).to(device)
    z = torch.randn((N, Z_DIM)).to(device)
    assert gen(enc(x),z).shape == (N, in_channels, H, W), "Generator test failed"
    print('Given GAN architecture passed the shape test')
    return True


def train_cnn(model, train_loader, optimizer, criterion, Epoch, log_interval, device):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval ==0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx*len(image), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
    return output

def evaluate_cnn(model, test_loader, criterion, device):
    model.eval()
    test_loss=0
    correct=0
    
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1,keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100*correct / len(test_loader.dataset)
    return test_loss, test_accuracy



def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    
    # Calculate critic scores
    scores_for_interpolated = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=scores_for_interpolated,
        grad_outputs=torch.ones_like(scores_for_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(state, filename="default"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gen, critic):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    critic.load_state_dict(checkpoint['critic'])

if __name__ == '__main__':
    print('my_utils.py is compiled.')