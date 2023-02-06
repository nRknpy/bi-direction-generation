import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_dim=28*28, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        #encoder
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        #decoder
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5= nn.Linear(h_dim, x_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


class ConvVAE(nn.Module):
    def __init__(self, x_dim=28*28, z_dim=20):
        super(ConvVAE, self).__init__()
        self.z_dim = z_dim
        
        #encoder
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(5*5*32, 256)
        self.fc2_1 = nn.Linear(256, z_dim)
        self.fc2_2 = nn.Linear(256, z_dim)
        
        #decoder
        self.fc4 = nn.Linear(z_dim, 400)
        self.fc5= nn.Linear(400, x_dim)
    
    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2_1(x), self.fc2_2(x)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
