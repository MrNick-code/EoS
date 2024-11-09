import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import data_man_pick_copy as dmp
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import cv2

out_features = 64

class cvae_function:
    
    def __init__(self, x, y):
        '''
        args
        - x ~ bayesian parameter v_n² 
        - y ~ particle spectrum rho(P_t, phi)
        '''
        self.x = x
        self.y = y
        self.encoder2d = self.conv2D_layers()
        self.encoderLin = self.common_lin_layers()
        self.q_phi = self.q_phi # no () cuz we want this to be
        self.r_theta1 = self.r_theta1 # functions and not tuples!
        self.r_theta2 = self.r_theta2
         
    def conv2D_layers(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8), padding=3),
            nn.Dropout(.2),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding=3),
            nn.Dropout(.2),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        
    def common_lin_layers(self):
        return nn.Sequential(
            nn.Linear(in_features=32, out_features=out_features), # 7
            nn.LeakyReLU(),
            nn.Linear(in_features=out_features, out_features=32), # 8
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=16), # 9
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
        
        
        in_features = xy.shape[1] # formating porpuses
        self.encoderLin[0] = nn.Linear(in_features=in_features, out_features=out_features)
        
        
        fc_out = self.encoderLin(xy) # apply fully conected layers
    
        layer10 = nn.Linear(in_features=16, out_features=3) # 10th layer
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
        
        in_features = flattened_y.shape[1] # formating porpuses
        self.encoderLin[0] = nn.Linear(in_features=in_features, out_features=out_features) 
        
        fc_out = self.encoderLin(flattened_y) # apply fully conected layers
        
        layer10 = nn.Linear(in_features=16, out_features=3) # 10th layer
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
        
        in_features = zy.shape[1] # formating porpuses
        self.encoderLin[0] = nn.Linear(in_features=in_features, out_features=out_features) 
        
        fc_out = self.encoderLin(zy)
        
        layer10 = nn.Linear(in_features=16, out_features=3) # 10th layer
        
        layer10a = nn.Sigmoid() # instantiating
        layer10b = nn.ReLU()
        
        mu_r2 = layer10a(layer10(fc_out))
        logvar_r2 = -layer10b(layer10(fc_out))       
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
    
    def latent_space_space(self, space_spec='', zn=None):
        '''
        args
        - zn ~ diferentiate r2theta function with input zq or zr (defaul 0)
          space_spec ~ define which function we're using
        '''

        #self.fc_mu = nn.Linear(in_features=self.flattened_size, out_features=3) # essas linhas são assim mesmo?
        #self.fc_logvar = nn.Linear(in_features=self.flattened_size, out_features=3)        
        '''
        desc
        - outputs the latent space using q_phi function
        
        
        desc
        - outputs the latent space using r_theta1 function

        returns
        - z_r ~ samples from the r_theta1 latent space representation
        
        
        desc
        - outputs the latent space using r_theta2 function

        returns
        - x_samp ~ samples from the r_theta2 latent space representation
        '''
        if space_spec == 'train':
            mu_q, logvar_q = self.q_phi()
            mu, logvar = mu_q, logvar_q
            z = self.reparameterize(mu_q, logvar_q) # z_q
        
        elif space_spec == 'test':
            mu_r, logvar_r = self.r_theta1()
            mu, logvar = mu_r, logvar_r
            z = self.reparameterize(mu_r, logvar_r) # z_r
        
        elif space_spec == 'output':
            
            if zn is None:
                raise ValueError('you must provide an correctly zn (zq or zr)')
            else:                  
                mu_r, logvar_r = self.r_theta2(zn) # was forgetting the r_theta2 input!
                    
                mu, logvar = mu_r, logvar_r
                z = self.reparameterize(mu_r, logvar_r) # x_samp
            
        else:
            print(f'Invalid space_spec value: {space_spec}')  
        return mu, logvar, z

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

# ---------------------------------------------------------------------------------

class cvae_load:
    
    def __init__(self, nt):
        self.nt = nt
        self.ld = self.loadDataNorm
        self.norm = self.normalize
        self.ri = self.resize_images

    def loadDataNorm(self, EOSXimgs_40, EOSXimgs_50):
        X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40 = dmp.get_data(paths=EOSXimgs_40, shape=(50, 50, 4))
        X_train40, X_val40, X_test40 = self.norm(X_train40, X_val40, X_test40, self.nt)
        xy40 = [X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40]
        
        X_train50, X_val50, X_test50, Y_train50, Y_val50, Y_test50 = dmp.get_data(paths=EOSXimgs_50, shape=(60, 60, 4))
        X_train50, X_val50, X_test50 = self.norm(X_train50, X_val50, X_test50, self.nt)
        xy50 = [X_train50, X_val50, X_test50, Y_train50, Y_val50, Y_test50]
        return xy40, xy50
    
    # Normalization function
    def normalize(self, train, val, test, normalization_term):
        return train / normalization_term, val / normalization_term, test / normalization_term

    # resize images shape to match input shape inside the network
    def resize_images(self, images, size):
        resized_images = []
        for img in images:
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            resized_images.append(resized_img)
        return np.array(resized_images)

    # [Num samples, Hidht, Weight, Chanells] --> [Num samples, Chanells, Hidht, Weight]
    def NCHW(self, trainX, valX, testX, trainY, valY, testY, trainX2, valX2, testX2, trainY2, valY2, testY2, cs, bs):

        trainX = self.ri(trainX, cs)
        valX = self.ri(valX, cs)
        testX = self.ri(testX, cs)

        trainX2 = self.ri(trainX2, cs)
        valX2 = self.ri(valX2, cs)
        testX2 = self.ri(testX2, cs)

        trainX = np.concatenate((trainX, trainX2), axis=0)
        valX = np.concatenate((valX, valX2), axis=0)
        testX = np.concatenate((testX, testX2), axis=0)
        trainY = np.concatenate((trainY, trainY2), axis=0)
        valY = np.concatenate((valY, valY2), axis=0)
        testY = np.concatenate((testY, testY2), axis=0)

        b = [torch.from_numpy(j).float() for j in [trainY, valY, testY]]
        a = [torch.from_numpy(i).float().permute(0, 3, 1, 2) for i in [trainX, valX, testX]]

        train_dataset = TensorDataset(a[0], b[0])
        val_dataset = TensorDataset(a[1], b[1])
        test_dataset = TensorDataset(a[2], b[2])


        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        return train_loader, val_loader, test_loader
