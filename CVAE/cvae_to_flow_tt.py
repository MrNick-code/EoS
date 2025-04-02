import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import data_man_pick_copy as dmp
import cv2
import numpy as np
from numpy import concatenate as concat
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

# cuda setup
device = torch.device("cuda")
torch.cuda.empty_cache()
# print(f'{torch.cuda.memory_summary(device=None, abbreviated=False)}') # to check cuda memory usage

# data paths
EOSXimgs_40 = ["C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSL_low_40b", "C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSQ_low_40"] # 50x50
EOSXimgs_50 = ["C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSL_low_50", "C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSQ_low_50"] # 60x60

# hyperparameters
common_size, batch_size, normalization_term_X, normalization_term_Y = (100, 100), 32, 1e-3, 255 # data set
out_features, features_multiplier = 64, 1                                                       # network
num_epochs, learning_rate = 300, 5e-5                                                           # fit
# (100, 100); 32; 255; 1e-3 / 64; (either 1 or 10) / 300; 5e-5 (last best run)

class cvae_load:
    
    def __init__(self, ntX, ntY):
        '''
        desc
        - transform data into pytorch necessities, load, normalize and resize it
        args
        - ntX ~ normalization term for |v_n|² (x)
          ntY ~ normalization term for pho(P_t, phi) (y)
        returns
        - data_loader ~ the loader of every dataset (train, validatino and test)
        '''
        self.ntX = ntX
        self.ntY = ntY
        self.ld = self.loadDataNorm
        self.norm = self.normalize
        self.ri = self.resize_images

    # load and normalize the data
    def loadDataNorm(self, EOSXimgs_40, EOSXimgs_50): # X and Y are swapped here in var names (ntx/y is right)
        X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40 = dmp.get_data(paths=EOSXimgs_40, shape=(50, 50, 4))
        # print("Unique values in X_train40 (y) before normalization:", np.unique(X_train40)) # images bin values are okay 
        X_train40, X_val40, X_test40 = self.norm(X_train40, X_val40, X_test40, self.ntY)
        Y_train40, Y_val40, Y_test40 = self.norm(Y_train40, Y_val40, Y_test40, self.ntX) # test because Adry norm report
        xy40 = [X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40]
        
        X_train50, X_val50, X_test50, Y_train50, Y_val50, Y_test50 = dmp.get_data(paths=EOSXimgs_50, shape=(60, 60, 4))
        X_train50, X_val50, X_test50 = self.norm(X_train50, X_val50, X_test50, self.ntY)
        Y_train50, Y_val50, Y_test50 = self.norm(Y_train50, Y_val50, Y_test50, self.ntX)
        xy50 = [X_train50, X_val50, X_test50, Y_train50, Y_val50, Y_test50]
        # print(X_train40.dtype) # X (y) is float64
        # print("Unique values in X_train40 (y) after normalization:", np.unique(X_train40))
        return xy40, xy50
    
    # normalization function
    def normalize(self, train, val, test, normalization_term):
        return train / normalization_term, val / normalization_term, test / normalization_term

    # resize images shape to match input shape inside the network
    def resize_images(self, images, size):
        resized_images = []
        for img in images:
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA) # correclty number of color channels
            resized_images.append(resized_img)
        return np.array(resized_images)

    # transform [Num samples, Hidht, Weight, Chanells] into [Num samples, Chanells, Hidht, Weight]
    def NCHW(self, trainY, valY, testY, trainX, valX, testX, trainY2, valY2, testY2, trainX2, valX2, testX2, cs, bs):

        trainY, valY, testY = self.ri(trainY, cs), self.ri(valY, cs), self.ri(testY, cs) # resize_image(y = pho(P_t, phi))

        trainY2, valY2, testY2 = self.ri(trainY2, cs), self.ri(valY2, cs), self.ri(testY2, cs)

        trainY, valY, testY = concat((trainY, trainY2), axis=0), concat((valY, valY2), axis=0), concat((testY, testY2), axis=0)

        trainX, valX, testX = concat((trainX, trainX2), axis=0), concat((valX, valX2), axis=0), concat((testX, testX2), axis=0)

        b = [torch.from_numpy(j).float() for j in [trainX, valX, testX]]
        a = [torch.from_numpy(i).float().permute(0, 3, 1, 2) for i in [trainY, valY, testY]]

        train_dataset, val_dataset, test_dataset = TensorDataset(a[0], b[0]), TensorDataset(a[1], b[1]), TensorDataset(a[2], b[2])

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True, pin_memory=True) # shuffle    ~ avoid local min 
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True, drop_last=True, pin_memory=True)     # drop_last  ~ avoid last batch 
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, drop_last=True, pin_memory=True)  # pin_memory ~ CUDA related

        return train_loader, val_loader, test_loader

# create data loaders
data_loader = cvae_load(ntX=normalization_term_X, ntY=normalization_term_Y)
xy40, xy50 = data_loader.loadDataNorm(EOSXimgs_40, EOSXimgs_50)
train_loader, val_loader, test_loader = data_loader.NCHW(
                xy40[0], xy40[1], xy40[2], xy40[3], xy40[4], xy40[5],
                xy50[0], xy50[1], xy50[2], xy50[3], xy50[4], xy50[5],  
                cs=common_size, bs=batch_size)

class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, x, y, out_features):
        '''
        desc
        - my CVAE implementation
        - "LAVA: Latent-Aided Variational Autoregressor" (lmao)
        '''
        super().__init__()
        self.x = x
        self.y = y
        self.out_features = out_features

        # condition convolutional-encoder 
        self.encoder2d = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8), padding=3),
            nn.Dropout(.2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding=3),
            nn.Dropout(.2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        # linear-encoder dynamic inputs 
        with torch.no_grad():
            dummy_x = torch.randn(1, *x.shape[1:]).to(device)
            dummy_y = torch.randn(1, *y.shape[1:]).to(device)
            dummy_y = dummy_y.to(next(self.encoder2d.parameters()).device)
            conv_out = self.encoder2d(dummy_y)
            flattened_size = conv_out.view(conv_out.size(0), -1).shape[1]       # view() reshapes a tensor by 'stretching' or
            flattened_x = dummy_x.view(dummy_x.size(0), -1).shape[1]            # 'squeezing' its elements into the shape you specify.
            in_features = flattened_size + flattened_x                        # correct dimension
        
        print(f'{flattened_size}, {flattened_x}')
        print(in_features)

        # linear-encoder q
        self.encoderLin = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features*features_multiplier),
            nn.BatchNorm1d(out_features*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=out_features*features_multiplier, out_features=32*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(.2),
            
            nn.Linear(in_features=32*features_multiplier, out_features=64*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=64*features_multiplier, out_features=32*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(.2),
            
            nn.Linear(in_features=32*features_multiplier, out_features=16*features_multiplier),
            nn.BatchNorm1d(16*features_multiplier),
            nn.LeakyReLU()
        )
        
        # linear-encoder r1 
        self.encoderLinAlt = nn.Sequential(
            nn.Linear(in_features=in_features-3, out_features=out_features*features_multiplier),
            nn.BatchNorm1d(out_features*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=out_features*features_multiplier, out_features=32*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(.2),
            
            nn.Linear(in_features=32*features_multiplier, out_features=64*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=64*features_multiplier, out_features=32*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(.2),
            
            nn.Linear(in_features=32*features_multiplier, out_features=16*features_multiplier),
            nn.BatchNorm1d(16*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # final encoder layer
        self.layer10 = nn.Linear(in_features=16*features_multiplier, out_features=3)

        # linear-decoder 
        self.decoderLin = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=16*features_multiplier),
            nn.BatchNorm1d(16*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=16*features_multiplier, out_features=32*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(.2),
            
            nn.Linear(in_features=32*features_multiplier, out_features=64*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=64*features_multiplier, out_features=32*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(.2),

            nn.Linear(in_features=32*features_multiplier, out_features=out_features*features_multiplier),
            nn.BatchNorm1d(out_features*features_multiplier),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # final decoder-layer (out_features has to be 3, so loss function can compare x_targets and x_samp)
        self.layer10dec =  nn.Linear(in_features=out_features*features_multiplier, out_features=3)

    def encode(self):
        '''
        desc
        - encoder function
            - q_phi(z|x, y) ~ "recognition" encoder network
            - r_theta1(z|y) ~ encoder network
        returns
        - mu_q, logvar_q   ~ q_phi encoder output latent space representation 
        - mu_r1, logvar_r1 ~ r_theta1 encoder output latent space representation  
        '''
        conv_out = self.encoder2d(self.y).to(device)                # inputs the condition through convolutional-encoder (q)
        flattened_size = conv_out.view(conv_out.size(0), -1) 
        flattened_x = self.x.view(self.x.size(0), -1).to(device)  
        #print(f'{flattened_size[1]}, {flattened_x[1]}')
        xy = torch.cat((flattened_size, flattened_x), dim=1)        # flatten layer (q)
        #print(xy)
        
        fc_out = self.encoderLin(xy)                                # flattened data through linear-encoder (q)
        mu_q = self.layer10(fc_out)                                 # z_q mean
        logvar_q = self.layer10(fc_out)                             # z_q log variance

        conv_out_r = self.encoder2d(self.y).to(device)              # inputs the condition through convolutional-encoder (r1)
        #print(conv_out_r.shape)
        flattened_y_r = conv_out_r.view(conv_out_r.size(0), -1)     # flatten layer (r1)
        #print(conv_out_r.shape)
        
        fc_out_r = self.encoderLinAlt(flattened_y_r)                # flattened data through linear-encoder (r1)
        mu_r1 = self.layer10(fc_out_r).to(device)                   # z_r1 mean
        logvar_r1 = self.layer10(fc_out_r).to(device)               # z_r1 log variance
        #print(mu_r1, logvar_r1)
        
        return mu_q, logvar_q, mu_r1, logvar_r1

    def reparameterize(self, mu, logvar):
        '''
        desc
        - reparameterize trick
        args
        - mu, logvar ~ mean and log variance
        returns
        - normal distribution with a certrain mean and standart deviation 
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_q):
        '''
        desc
        - decoder function
            - r_theta2(x|y, z) ~ decoder network
        args
        - z_q ~ latent distribution generated from the q encoder network
        returns
        - mu_r2, logvar_r2 ~ r_theta2 decoder output latent space representation
        - z_r2             ~ N(mu_r2, logvar_r2²)
        '''
        conv_out = self.encoder2d(self.y).to(device)                # inputs the condition through convolutional-encoder (q)
        flattened_y = conv_out.view(conv_out.size(0), -1)           # gets flattened condition to use with latent space (z_q + c)
        #print(flattened_y[1])
        #print(z_q[1])
        zy = torch.cat((flattened_y, z_q), dim=1)                   # conditioned latent space (z_q + c)
        #print(f'\033[31;1m{zy}\033[m')
        
        fc_out = self.decoderLin(zy)                                # coditioned latent space through linear-decoder
        mu_r2 = torch.sigmoid(self.layer10dec(fc_out))              # ?????????
        logvar_r2 = -F.leaky_relu(self.layer10dec(fc_out))          # ?????????

        return mu_r2, logvar_r2

    def forward(self):
        mu, logvar, mur1, logvarr1 = self.encode()                                  # encode
        z = self.reparameterize(mu, logvar)                                         # latent space from encode
        #print(f'\033[31;1m{z}\033[m') # interesting print to analize z
        mur2, logvarr2 = self.decode(z)                                             # decode
        recon_data = self.reparameterize(mur2, logvarr2) # z_r2                     # latent space from decode
        return recon_data, mu, logvar, mur1, logvarr1, mur2, logvarr2

train_y, train_x = next(iter(train_loader))
train_y, train_x = train_y.to(device), train_x.to(device)

model = ConditionalVariationalAutoEncoder(train_x, train_y, out_features).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.2)
# Congelar os outros parâmetros para evitar acúmulo de gradiente
for param in model.parameters():
    param.requires_grad = False

# Liberar apenas os pesos do encoderLin e decoderLin
for param in model.encoderLin.parameters():
    param.requires_grad = True
for param in model.decoderLin.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.encoderLin.parameters()},
    {'params': model.decoderLin.parameters()}
], lr=learning_rate)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')  
        nn.init.zeros_(m.bias)

model.apply(init_weights)

train_loss_history, train_r2_history, val_loss_history, val_r2_history = [], [], [], []

#def cvae_loss_function(mu_q, logvar_q, mu_r1, logvar_r1, x_targets, x_samp):
'''
    args
    - mu_q      ~ mean from the encoder q_phi(z|x, y)
    - logvar_q  ~ log variance from the encoder q_phi(z|x, y)
    - mu_r1     ~ mean from the prior r_theta1(z|y)
    - logvar_r1 ~ log variance from the prior r_theta1(z|y)
    - x         ~ true data mean (ground truth)
    - x_samp    ~ reconstructed data from the decoder
    
    returns
    - total loss (reconstruction loss + KL divergence)
    '''
    #print(f'x = {x}')
    #print(f'y = {y}')
    #x = torch.cat((x, y), dim=0)
    #print(f"Input shape (x): {len(x)}")
    #print(f"Reconstructed shape (recon_data): {x_samp}")
    
    
'''x = x[:, :3, :, :].mean(dim=[2, 3])  ################### Solução provavelmente inviável
    if x.shape[0] != x_samp.shape[0]:
        min_batch = min(x.shape[0], x_samp.shape[0])
        x = x[:min_batch]  # Ajusta batch de x
        x_samp = x_samp[:min_batch]  # Ajusta batch de x_samp
    print(f"Input shape (x): {x.shape}")'''
    
#    recon_loss = F.mse_loss(x_samp, x_targets, reduction='sum') # reconstruction loss TA AQUI O PROBLEMA // resolvido com drop_last=True

    # KL-divergence loss between two gaussians: q_phi(z|x,y) & r_theta1(z|y)
#    kl_div = -0.5 * torch.sum(1 + logvar_q - logvar_r1 - ((mu_q - mu_r1).pow(2) + logvar_q.exp()) / logvar_r1.exp())
#    return recon_loss + kl_div

class cvae_loss_function(nn.Module): # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
    def __init__(self):
        super(cvae_loss_function,self).__init__()
    
    def forward(self, mu_q, logvar_q, mu_r1, logvar_r1, x_targets, x_samp, beta=0.1):
        recon_loss = F.smooth_l1_loss(x_samp, x_targets, reduction='sum')
        # recon_loss = F.mse_loss(x_samp, x_targets, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar_q - logvar_r1 - ((mu_q - mu_r1).pow(2) + logvar_q.exp()) / logvar_r1.exp())
        return recon_loss + beta * kl_div
    
#def calculate_r2(predictions, targets):
#    '''
#    args
#    - predictions ~ predictions from CVAE model
#    = targets     ~ actually targets we're aiming at
#    
#    returns
#    - r squared metric for analysis
#    '''
#    predictions_np = predictions.view(-1).detach().cpu().numpy()
#    targets_np = targets.view(-1).detach().cpu().numpy()
#    return r2_score(targets_np, predictions_np)

def calculate_r2(predictions, targets):
    predictions = predictions.view(-1).detach().cpu()
    targets = targets.view(-1).detach().cpu()

    # Evita NaN removendo valores inválidos
    mask = ~torch.isnan(predictions) & ~torch.isnan(targets)
    if mask.sum() == 0:
        return float('nan')  # Evita erro se todos forem NaN
    
    predictions = predictions[mask]
    targets = targets[mask]

    # Cálculo manual de R²
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)

    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

# -------------------------------------------------------------------------------------
'''bruh = cvae_loss_function()

for epoch in range(num_epochs):
    train_total_loss = 0.0
    train_epoch_r2 = 0.0
    val_total_loss = 0.0
    val_epoch_r2 = 0.0
    beta = min(0.1, epoch / 50)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    torch.cuda.empty_cache()

    model.train()
    for y, x in train_loader:
        
        y, x = y.to(device), x.to(device)
        #print(f"inputs size (y): {y.size()}")
        #print(f"targets size (x): {x.size()}") # I'M SO DUMB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        
        recon_data, mu_q, logvar_q, mu_r1, logvar_r1, mu_r2, logvar_r2 = model()       
        optimizer.zero_grad()        
        #print(model.encoderLin[0].weight.grad)
        #for param_group in optimizer.param_groups:
        #    print(param_group['params'])
        
        #print(f"x_samp size (output): {recon_data.size()}")
        #print(f"Tipo de recon_data: {type(recon_data)}")
        loss = bruh.forward(mu_q, logvar_q, mu_r1, logvar_r1, x, recon_data, beta=0.1) # lembrete: eu reconstruo x!
        rsq = calculate_r2(recon_data, x)   
        #print(f"TotalLoss: {loss}")
        
        #print(f"Loss antes do backward: {TotalLoss.item()}")
        #before_update = model.encoderLin[0].weight.clone()
        #print("Pesos antes da atualização:", network.encoderLin[0].weight)
        # print("Gradiente de encoderLin:", network.encoderLin[0].weight.grad)
        loss.backward()
        #print(model.encoderLin[0].weight.grad)
        
        #for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        print(name, param.grad.norm().item())

        
        train_total_loss += loss.detach().cpu().numpy()
        train_epoch_r2 += rsq * x.size(0)
        
        optimizer.step() 
        #after_update = model.encoderLin[0].weight.clone()
        #print(f"Máxima diferença nos pesos: {(after_update - before_update).abs().max()}")
        #print("Pesos após a atualização:", network.encoderLin[0].weight)
        #print(f"Pesos mudaram: {not torch.equal(before_update, after_update)}")
        #for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        print(f"Gradiente de {name}: {param.grad.abs().mean().item()}")
        
        #for param_group in optimizer.param_groups:
        #    for param in param_group['params']:
        #        print(param.shape, param.requires_grad)

        #for param in model.encoderLin.parameters():
        #    print(param.requires_grad)
        
        #print(model.encoderLin[0].weight.grad.mean())
        
        #found = False
        #for param_group in optimizer.param_groups:
        #    for param in param_group["params"]:
        #        if torch.equal(param.data, model.encoderLin[0].weight.data):
        #            found = True
        #print(f"encoderLin está no otimizador? {found}")



    #print(f"Input shape: {inputs.shape}")
    #print(f"Target shape: {targets.shape}")
    #print(f"Predicted shape: {x_samp.shape}")
    
    print(torch.mean(model.layer10.weight).item(), torch.std(model.layer10.weight).item())

    
    train_epoch_loss = train_total_loss / len(train_loader.dataset)
    train_epoch_r2 /= len(train_loader.dataset)
    
    train_loss_history.append(train_epoch_loss)
    train_r2_history.append(train_epoch_r2)
    
    model_val.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        for y_val, x_val in val_loader:
            y_val, x_val = y_val.to(device), x_val.to(device)
            x_val = x_val.view(-1, 3)
            
            recon_data_val, mu_q_val, logvar_q_val, mu_r1_val, logvar_r1_val, mu_r2_val, logvar_r2_val = model_val()     
    
            Valloss = bruh.forward(mu_q_val, logvar_q_val, mu_r1_val, logvar_r1_val, x_val, recon_data_val, beta=0.1)  
            Valrsq = calculate_r2(recon_data, x) 
            
            val_total_loss += Valloss.detach().cpu().numpy()
            val_epoch_r2 += Valrsq * x.size(0)
            
    print(f'x = {x[0:3]}')
    print(f"recon_data: {recon_data[0:3]}")
    
    val_epoch_loss = val_total_loss / len(val_loader.dataset)
    val_epoch_r2 /= len(val_loader.dataset)
    
    val_loss_history.append(val_epoch_loss)
    val_r2_history.append(val_epoch_r2)

    print(f"\033[31;1mEpoch {epoch + 1}/{num_epochs}\033[m: Train Loss={train_epoch_loss:.4f}, Train R²={train_epoch_r2:.4f}, "
        f"Val Loss={val_epoch_loss:.4f}, Val R²={val_epoch_r2:.4f}")    
'''
scaler = GradScaler()  # Inicializa o escalador de gradiente para AMP
bruh = cvae_loss_function()

for epoch in range(num_epochs):
    train_total_loss = 0.0
    train_epoch_r2 = 0.0
    val_total_loss = 0.0
    val_epoch_r2 = 0.0
    beta = min(0.1, epoch / 50)
    
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #torch.cuda.empty_cache()

    model.train()
    for y, x in train_loader:
        y, x = y.to(device), x.to(device)
        
        optimizer.zero_grad()

        with autocast(device_type="cuda"):  # Mixed precision ativado
            recon_data, mu_q, logvar_q, mu_r1, logvar_r1, mu_r2, logvar_r2 = model()       
            loss = bruh.forward(mu_q, logvar_q, mu_r1, logvar_r1, x, recon_data, beta=0.1)  
            rsq = calculate_r2(recon_data, x)   
        
        scaler.scale(loss).backward()  # Escala os gradientes para evitar underflow
        scaler.step(optimizer)  # Aplica o passo do otimizador
        scaler.update()  # Atualiza o escalador
        
        train_total_loss += loss.detach().cpu().numpy()
        train_epoch_r2 += rsq * x.size(0)
        
    train_epoch_loss = train_total_loss / len(train_loader.dataset)
    train_epoch_r2 /= len(train_loader.dataset)
    
    train_loss_history.append(train_epoch_loss)
    train_r2_history.append(train_epoch_r2)
    
    model.eval()
    with torch.no_grad(), autocast(device_type="cuda"):  # AMP no modo de validação
        #torch.cuda.empty_cache()
        for y_val, x_val in val_loader:
            y_val, x_val = y_val.to(device), x_val.to(device)
            x_val = x_val.view(-1, 3)
            
            recon_data_val, mu_q_val, logvar_q_val, mu_r1_val, logvar_r1_val, mu_r2_val, logvar_r2_val = model()
            Valloss = bruh.forward(mu_q_val, logvar_q_val, mu_r1_val, logvar_r1_val, x_val, recon_data_val, beta=0.1)  
            Valrsq = calculate_r2(recon_data_val, x_val) 
            
            val_total_loss += Valloss.detach().cpu().numpy()
            val_epoch_r2 += Valrsq * x.size(0)
    
    val_epoch_loss = val_total_loss / len(val_loader.dataset)
    val_epoch_r2 /= len(val_loader.dataset)
    
    val_loss_history.append(val_epoch_loss)
    val_r2_history.append(val_epoch_r2)

    print(f"\033[31;1mEpoch {epoch + 1}/{num_epochs}\033[m: Train Loss={train_epoch_loss:.4f}, Train R²={train_epoch_r2:.4f}, "
          f"Val Loss={val_epoch_loss:.4f}, Val R²={val_epoch_r2:.4f}")
    
    if epoch % 5 == 0:
        print(f"x: {x[0].cpu().detach().numpy()[:5]}")
        print(f"Recon: {recon_data[0].cpu().detach().numpy()[:5]}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm().item()}")




# Data Analysis
# plot loss and R^2 curves
plt.figure(figsize=(12, 5))

# removing extreme outlier
train_loss_history_mod = train_loss_history[1:]
val_loss_history_mod = val_loss_history[1:]
train_r2_history_mod = train_r2_history[1:]
val_r2_history_mod = val_r2_history[1:]
epochs_mod = range(2, num_epochs + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs_mod, train_loss_history_mod, label='Train Loss')
plt.plot(epochs_mod, val_loss_history_mod, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
# if r² is negative, that means - the model is worst then outputing the mean value.
plt.plot(epochs_mod, train_r2_history_mod, label='Train $R²$')
plt.plot(epochs_mod, val_r2_history_mod, label='Val $R²$')
plt.xlabel('Epochs')
plt.ylabel('$R²$')
plt.legend()

plt.show()


model.eval()
predictions, real_values = [], []
first_iteration = True

with torch.no_grad():
    for y, x in test_loader:
        recon_data, mu_q, logvar_q, mu_r1, logvar_r1, mu_r2, logvar_r2 = model()

        predictions.append(recon_data)  # Y^ (in this case now, it's acttualy X^)
        real_values.append(x)  # Y (and here is X)

# Single arrays
predictions = torch.cat(predictions, dim=0).cpu().numpy()
real_values = torch.cat(real_values, dim=0).cpu().numpy()

print(f'real_value shape: {real_values.shape}')
print(f'prediction shape: {predictions.shape}')

print(f'\033[32;1mV2² test R²: {r2_score(real_values[0], recon_data[0].cpu().numpy())}\033[m')
print(f'\033[32;1mV3² test R²: {r2_score(real_values[1], recon_data[1].cpu().numpy())}\033[m')
print(f'\033[32;1mV4² test R²: {r2_score(real_values[2], recon_data[2].cpu().numpy())}\033[m')

# scatter plot real vs predictions v2², v3² & v4²
predictions = predictions[:, :3]
real_values = real_values[:, :3]

fig, axs = plt.subplots(1, 3, figsize=(12, 5))

for i in range(3):
    axs[i].scatter(real_values[:, i], predictions[:, i], c='seagreen', label=f'v_{i+2}²')
    axs[i].plot(real_values[:, i], real_values[:, i], color='lime', linestyle='--') # Reference line
    axs[i].set_xlabel(f'Real v_{i+2}²')
    axs[i].set_ylabel(f'Predicted v_{i+2}²')
    axs[i].legend()

plt.suptitle('Real vs. Predicted Values for Each Output Dimension')
plt.show()
