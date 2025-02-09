import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import data_man_pick_copy as dmp
import cv2
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

# cuda setup
device = torch.device("cuda")

# hyper params
out_features = 64

EOSXimgs_40 = ["C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSL_low_40b", "C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSQ_low_40"] # Path (50x50 imgs)
EOSXimgs_50 = ["C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSL_low_50", "C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSQ_low_50"] # Path (60x60 imgs)

common_size, batch_size, num_epochs, learning_rate, normalization_term = (50, 50), 8, 10, 2e-5, 255

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

data_loader = cvae_load(nt=normalization_term)

xy40, xy50 = data_loader.loadDataNorm(EOSXimgs_40, EOSXimgs_50)

train_loader, val_loader, test_loader = data_loader.NCHW(
                xy40[0], xy40[1], xy40[2], xy40[3], xy40[4], xy40[5],
                xy50[0], xy50[1], xy50[2], xy50[3], xy50[4], xy50[5],  
                cs=common_size, bs=batch_size)

class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.x = x.to(device)
        self.y = y.to(device)
        self.encoder2d = self.conv2D_layers()
        self.encoderLin = self.common_lin_layers()
    
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
    
    def encode(self):
        '''
        desc
        - q_phi(z|x, y) ~ "recognition" encoder network
          where,
            phi ~ trainable parameter set
            z   ~ locations within a latent space
        returns
        - mu_q, logvar_q  ~ q_phi encoder output latent space representation
        '''
        # conv_out = self.encoder(torch.randn(1, 3, 1000)) # test network and calculates output itermidiate dimension (dummy input)
        
        #y = y.unsqueeze(-1).unsqueeze(-1)  # Transforma [batch_size, 3] em [batch_size, 3, 1, 1]

        
        conv_out = self.encoder2d(self.y).to(device) # apply conv layers        
        flattened_size = conv_out.view(conv_out.size(0), -1).to(device) # flatten layer (keeping batch size untouched)
        flattened_x = self.x.view(self.x.size(0), -1).to(device) # flatten the vn² too      
        xy = torch.cat((flattened_size, flattened_x), dim=1).to(device) # append(x)        
        #print(f"Self encoder Lin: {self.encoderLin}")
        in_features = xy.shape[1] # formating porpuses
        self.encoderLin[0] = nn.Linear(in_features=in_features, out_features=out_features).to(device)    
        fc_out = self.encoderLin(xy).to(device) # apply fully conected layers   
        layer10 = nn.Linear(in_features=16, out_features=3).to(device) # 10th layer
        mu_q = layer10(fc_out).to(device)
        logvar_q = layer10(fc_out).to(device)
        
        '''
        desc
        - r_theta1(z|y) ~ encoder network
          where, 
            theta ~ trainable neural network parameters sets
            
        returns
        - mu_r1, logvar_r1 ~ r_theta1 encoder output latent space representation
        '''
        conv_out_r = self.encoder2d(self.y).to(device) # apply conv layers
        flattened_y_r = conv_out_r.view(conv_out.size(0), -1).to(device) # flatten layer
        in_features_r = flattened_y_r.shape[1] # formating porpuses
        self.encoderLin[0] = nn.Linear(in_features=in_features_r, out_features=out_features).to(device) 
        fc_out_r = self.encoderLin(flattened_y_r).to(device) # apply fully conected layers
        layer10_r = nn.Linear(in_features=16, out_features=3).to(device) # 10th layer
        mu_r1 = layer10_r(fc_out_r).to(device)
        logvar_r1 = layer10_r(fc_out_r).to(device)
        
        return mu_q, logvar_q, mu_r1, logvar_r1
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z_q):
        '''
        desc
        - r_theta2(x|y, z) ~ decoder network
        args
        - z_q ~ samples from the q_phi latent space representation
        returns
        - mu_r2, logvar_r2 ~ the output of the decoder (a distribution in the physical parameter space)
        '''
        conv_out = self.encoder2d(self.y).to(device)
        flattened_y = conv_out.view(conv_out.size(0), -1).to(device)
        zy = torch.cat((flattened_y, z_q.to(device)), dim=1).to(device) # append(z)      
        in_features = zy.shape[1] # formating porpuses
        self.encoderLin[0] = nn.Linear(in_features=in_features, out_features=out_features).to(device)     
        fc_out = self.encoderLin(zy).to(device)
        layer10 = nn.Linear(in_features=16, out_features=3).to(device) # 10th layer
        layer10a = nn.Sigmoid().to(device) # instantiating
        layer10b = nn.ReLU().to(device)
        mu_r2 = layer10a(layer10(fc_out)).to(device)
        logvar_r2 = -layer10b(layer10(fc_out)).to(device)
        
         # Reparameterization trick to sample from the distribution
        std_r2 = torch.exp(0.5 * logvar_r2)  # Standard deviation
        eps = torch.randn_like(std_r2)  # Random noise
        z_r2 = mu_r2 + eps * std_r2  # Sample from the distribution
            
        return z_r2
    
    def forward(self):
        mu, logvar, mur1, logvarr1 = self.encode()
        z = self.reparameterize(mu, logvar)
        recon_data = self.decode(z)
        return recon_data, mu, logvar, mur1, logvarr1

train_y, train_x = next(iter(train_loader))
train_y, train_x = train_y.to(device), train_x.to(device)

model = ConditionalVariationalAutoEncoder(train_x, train_y).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_history, train_r2_history, val_loss_history, val_r2_history = [], [], [], []

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
    print(f"Input shape (x): {x.shape}")
    print(f"Reconstructed shape (recon_data): {x_samp.shape}")
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

# -------------------------------------------------------------------------------------

for epoch in range(num_epochs):
    train_total_loss = 0.0
    train_epoch_r2 = 0.0
    
    model.train()
    for inputs, targets in train_loader:
        
        inputs, targets = inputs.to(device), targets.to(device)
        #print(f"inputs size (x): {inputs.size()}")
        #print(f"targets size (y): {targets.size()}") 
        
        recon_data, mu_q, logvar_q, mu_r1, logvar_r1 = model()       
        optimizer.zero_grad()        
        #print(network.encoderLin[0].weight.grad)
        #for param_group in optimizer.param_groups:
        #    print(param_group['params'])
        
        #print(f"x_samp size (output): {recon_data.size()}")
        print(f"Tipo de recon_data: {type(recon_data)}")
        loss = cvae_loss_function(mu_q, logvar_q, mu_r1, logvar_r1, inputs, recon_data)  
        rsq = calculate_r2(recon_data, targets)   
        #print(f"TotalLoss: {loss}")
        
        #print(f"Loss antes do backward: {TotalLoss.item()}")
        #before_update = network.encoderLin[0].weight.clone()
        #print("Pesos antes da atualização:", network.encoderLin[0].weight)
        # print("Gradiente de encoderLin:", network.encoderLin[0].weight.grad)
        loss.backward()
        train_total_loss += loss.detach().cpu().numpy()
        train_epoch_r2 += rsq * inputs.size(0)
        
        optimizer.step() 
        #after_update = network.encoderLin[0].weight.clone()
        #print("Pesos após a atualização:", network.encoderLin[0].weight)
        #print(f"Pesos mudaram: {not torch.equal(before_update, after_update)}")
        #for name, param in network.named_parameters():
        #    if param.grad is not None:
        #        print(f"Gradiente de {name}: {param.grad.abs().mean().item()}")
        

    #print(f"Input shape: {inputs.shape}")
    #print(f"Target shape: {targets.shape}")
    #print(f"Predicted shape: {x_samp.shape}")
    
    train_epoch_loss = train_total_loss / len(train_loader.dataset)
    train_epoch_r2 /= len(train_loader.dataset)
    
    train_loss_history.append(train_epoch_loss)
    train_r2_history.append(train_epoch_r2)
    
    val_total_loss = 0.0
    val_epoch_r2 = 0.0
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 3)
            
            recon_data, mu_q, logvar_q, mu_r1, logvar_r1 = model() 
            
            Valloss = cvae_loss_function(mu_q, logvar_q, mu_r1, logvar_r1, inputs, recon_data)  
            Valrsq = calculate_r2(recon_data, targets) 
            
            val_total_loss += Valloss.detach().cpu().numpy()
            val_epoch_r2 += Valrsq * inputs.size(0)
            
    #print(f"Validation Input shape: {inputs.shape}")
    #print(f"Validation Target shape: {targets.shape}")
    #print(f"Validation Predicted shape: {x_samp.shape}")
    
    val_epoch_loss = val_total_loss / len(val_loader.dataset)
    val_epoch_r2 /= len(val_loader.dataset)
    
    val_loss_history.append(val_epoch_loss)
    val_r2_history.append(val_epoch_r2)

    print(f"\033[31;1mEpoch {epoch + 1}/{num_epochs}\033[m: Train Loss={train_epoch_loss}, Train R²={train_epoch_r2:.4f}, "
        f"Val Loss={val_epoch_loss}, Val R²={val_epoch_r2:.4f}")    
