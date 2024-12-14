import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class ConvGenerator(nn.Module):
    def __init__(self, g_input_dim, g_output_channels):
        super(ConvGenerator, self).__init__()

        self.g_hidden = 64

        self.conv1 = nn.ConvTranspose2d(g_input_dim, self.g_hidden * 8, 4, 1, 0, bias=False)
        self.batch1 = nn.BatchNorm2d(self.g_hidden * 8)

        self.conv2 = nn.ConvTranspose2d(self.g_hidden * 8, self.g_hidden * 4, 4, 2, 1, bias=False)
        self.batch2 = nn.BatchNorm2d(self.g_hidden * 4)

        self.conv3 = nn.ConvTranspose2d(self.g_hidden * 4, self.g_hidden * 2, 4, 2, 1, bias=False)
        self.batch3 = nn.BatchNorm2d(self.g_hidden * 2)

        self.conv4 = nn.ConvTranspose2d(self.g_hidden * 2, self.g_hidden, 4, 2, 1, bias=False)
        self.batch4 = nn.BatchNorm2d(self.g_hidden)

        self.conv5 = nn.ConvTranspose2d(self.g_hidden, g_output_channels, 4, 2, 1, bias=False)
    
    # forward method
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.batch3(self.conv3(x)))
        x = F.relu(self.batch4(self.conv4(x)))

        return torch.tanh(self.conv5(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))
    
class ConvDiscriminator(nn.Module):
    def __init__(self, d_input_channels):
        super(ConvDiscriminator, self).__init__()

        self.d_hidden = 64

        self.conv1 = nn.Conv2d(d_input_channels, self.d_hidden, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(self.d_hidden, self.d_hidden * 2, 4, 2, 1, bias=False)
        self.batch2 = nn.BatchNorm2d(self.d_hidden * 2)

        self.conv3 = nn.Conv2d(self.d_hidden * 2, self.d_hidden * 4, 4, 2, 1, bias=False)
        self.batch3 = nn.BatchNorm2d(self.d_hidden * 4)

        self.conv4 = nn.Conv2d(self.d_hidden * 4, self.d_hidden * 8, 4, 2, 1, bias=False)
        self.batch4 = nn.BatchNorm2d(self.d_hidden * 8)

        self.conv5 = nn.Conv2d(self.d_hidden * 8, 1, 4, 1, 0, bias=False)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.batch2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.batch3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.batch4(self.conv4(x)), 0.2)

        return torch.sigmoid(self.conv5(x)).view(-1, 1)
    
def D_train(D, G, D_optimizer, mnist_dim, z_dim, bs, device, criterion, x, use_conv=False):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real data
    if use_conv:
        x_real, y_real = x, torch.ones(bs, 1)
    else:
        x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    # train discriminator on fake data
    z = Variable(torch.randn(bs, z_dim).to(device))
    if use_conv:
        z = z.view(bs, z_dim, 1, 1)
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def G_train(D, G, G_optimizer, mnist_dim, z_dim, bs, device, criterion, x, use_conv=False):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    if use_conv:
        z = z.view(bs, z_dim, 1, 1)
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()