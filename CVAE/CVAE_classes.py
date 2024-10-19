import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

class cvae_function:
    
    def __init__(self, x, y):
        '''
        args
        - x ~ bayesian parameter v_n² 
        - y ~ particle spectrum pho(P_t, phi)
        '''
        self.x = x
        self.y = y
        self.encoder2d = self.conv2D_layers()
        self.encoderLin = self.common_lin_layers()
        self.reparameterize = self.reparameterize()
        self.q_phi = self.q_phi()
        self.r_theta1 = self.r_theta1()
        self.r_theta2 = self.r_theta2()
         
    def conv2D_layers(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=64), # 1
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=32, stride=4), # 2
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=32), # 3
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=16, stride=2), # 4
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=16), # 5
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=16, stride=2), # 6
            nn.LeakyReLU()
        )
        
    def common_lin_layers(self):
        return nn.Sequential(
            nn.Linear(in_features=6159, out_features=4096), # 7
            nn.LeakyReLU(),
            nn.Linear(in_features=4096, out_features=2048), # 8
            nn.LeakyReLU(),
            nn.Linear(in_features=2048, out_features=1024), # 9
            nn.LeakyReLU()
        )
        
    def q_phi(self):
        '''
        desc
        - q_phi(z|, x, y) ~ "recognition" encoder network
          where,
            phi ~ trainable parameter set
            z   ~ locations within a latent space
        returns
        - mu_q, logvar_q  ~ q_phi encoder output latent space representation
        '''
        # conv_out = self.encoder(torch.randn(1, 3, 1000)) # test network and calculates output itermidiate dimension (dummy input)
        conv_out = self.encoder2d(self.y) # apply conv layers
        
        flattened_size = conv_out.view(conv_out.size(0), -1) # flatten layer (keeping batch size untouched)
        flattened_x = self.x.view(self.x.size(0), -1) # flatten the vn² too
        
        xy = torch.cat((flattened_size, flattened_x), dim=1) # append(x)
        
        fc_out = self.encoderLin(xy) # apply fully conected layers
    
        layer10 = nn.Linear(in_features=1024, out_features=30) # 10th layer
        mu_q = layer10(fc_out)
        logvar_q = layer10(fc_out)
        return mu_q, logvar_q

    def r_theta1(self):
        '''
        desc
        - r_theta1(z|y) ~ encoder network
          where, 
            theta ~ trainable neural network parameters sets
            
        returns
        - mu_r1, logvar_r1 ~ r_theta1 encoder output latent space representation
        '''
        conv_out = self.encoder2d(self.y) # apply conv layers
        
        flattened_y = conv_out.view(conv_out.size(0), -1) # flatten layer
        
        fc_out = self.encoderLin(flattened_y) # apply fully conected layers
        
        layer10 = nn.Linear(in_features=1024, out_features=960) # 10th layer
        mu_r1 = layer10(fc_out)
        logvar_r1 = layer10(fc_out)       
        return mu_r1, logvar_r1

    def r_theta2(self, z_q):
        '''
        desc
        - r_theta2(x|, y, z) ~ decoder network
        args
        - z_q ~ samples from the q_phi latent space representation
        returns
        - mu_r2, logvar_r2 ~ the output of the decoder (a distribution in the physical parameter space)
        '''
        conv_out = self.encoder2d(self.y)
        
        flattened_y = conv_out.view(conv_out.size(0), -1)
        
        zy = torch.cat((flattened_y, z_q), dim=1) # append(z)
        
        fc_out = self.encoderLin(zy)
        
        layer10 = nn.Linear(in_features=1024, out_features=30) # 10th layer
        layer10a = nn.Sigmoid(layer10)
        layer10b = nn.ReLU(layer10)
        mu_r2 = layer10a(fc_out)
        logvar_r2 = -layer10b(fc_out)       
        return mu_r2, logvar_r2

    def reparameterize(self, mu, logvar):
        '''
        desc
        - builds encoders output latent space distributions
        
        args
        - mu, logvar = latent space dimensions
        
        returns
        - gaussian space representation
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)        
        return mu + eps * std
    
    def latent_space_space(self, space_spec=''):

        self.fc_mu = nn.Linear(in_features=self.flattened_size, out_features=3)
        self.fc_logvar = nn.Linear(in_features=self.flattened_size, out_features=3)        
        '''
        desc
        - outputs the latent space using q_phi function
        '''
        if space_spec == 'train':
            mu_q, logvar_q = self.q_phi()
            z = self.reparameterize(mu_q, logvar_q) # z_q
        
        '''
        desc
        - outputs the latent space using r_theta1 function

        returns
        - z_r ~ samples from the r_theta1 latent space representation
        '''
        if space_spec == 'test':
            mu_r, logvar_r = self.r_theta1()
            z = self.reparameterize(mu_r, logvar_r) # z_r
        
        '''
        desc
        - outputs the latent space using r_theta2 function

        returns
        - x_samp ~ samples from the r_theta2 latent space representation
        ''' 
        if space_spec == 'output':
            mu_r, logvar_r = self.r_theta2()
            z = self.reparameterize(mu_r, logvar_r) # x_samp
            
        else:
            print('space_spec argument does not exist!')     
        return z

# ---------------------------------------------------------------------------------

def cvae_loss_function(mu_q, logvar_q, mu_r1, logvar_r1, x, x_samp):
    '''
    args
    - mu_q      ~ mean from the encoder q_phi(z|x, y)
    - logvar_q  ~ log variance from the encoder q_phi(z|x, y)
    - mu_r1     ~ mean from the prior r_theta1(z|y)
    - logvar_r1 ~ log variance from the prior r_theta1(z|y)
    - x         ~ true data mean (ground truth)
    - x_samp ~ reconstructed data from the decoder
    
    returns
    - total loss (reconstruction loss + KL divergence)
    '''
    recon_loss = F.mse_loss(x_samp, x, reduction='sum') # reconstruction loss

    # KL-divergence loss between two gaussians: q_phi(z|x,y) & r_theta1(z|y)
    kl_div = -0.5 * torch.sum(1 + logvar_q - logvar_r1 - ((mu_q - mu_r1).pow(2) + logvar_q.exp()) / logvar_r1.exp())
    return recon_loss + kl_div

def calculate_r2(predictions, targets):
    '''
    args
    - predictions ~ predictions from CVAE model
    = targets     ~ actually targets we're aiming at
    
    returns
    - r squared metric for analysis
    '''
    predictions_np = predictions.view(-1).detach().cpu().numpy()
    targets_np = targets.view(-1).detach().cpu().numpy()
    return r2_score(targets_np, predictions_np)
