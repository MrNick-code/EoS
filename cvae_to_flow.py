import torch
import torch.nn as nn
import CVAE_classes as cc

EOSXimgs_40 = ["C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSL_low_40b", "C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSQ_low_40"] # Path (50x50 imgs)
EOSXimgs_50 = ["C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSL_low_50", "C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSQ_low_50"] # Path (60x60 imgs)

common_size, batch_size, num_epochs, learning_rate, normalization_term = (50, 50), 8, 50, 2e-5, 255

data_loader = cc.cvae_load(nt=normalization_term)

xy40, xy50 = data_loader.loadDataNorm(EOSXimgs_40, EOSXimgs_50)

train_loader, val_loader, test_loader = data_loader.NCHW(
                xy40[0], xy40[1], xy40[2], xy40[3], xy40[4], xy40[5],
                xy50[0], xy50[1], xy50[2], xy50[3], xy50[4], xy50[5],  
                cs=common_size, bs=batch_size)

class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, x1, x2, x3, y1, y2, y3): # train_loader[0], val_loader[0], test_loader[0], train_loader[1], val_loader[1], test_loader[1]
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        
        ## Aqui, inicializamos os componentes que precisam ser usados nos métodos da rede
        #self.encoder2d = cc.cvae_function.conv2D_layers(self)  # Criação das camadas convolucionais
        #self.encoderLin = cc.cvae_function.common_lin_layers(self)  # Camadas lineares comuns
        ## Certifique-se de definir os métodos adicionais que fazem parte do modelo, se necessário

    
    def doFunc(self, spec=''):
        '''
        desc
        - chew CVAE_classes.py
        
        args
        - space_spec object
        
        returns
        - z_q    ~ training(a)/validation(a2) latent space
          z_r    ~ test latent space
          x_samp ~ predicted values
          H      ~ training(a)/validation(a2) total loss
          rsq    ~ r squared metric
        '''
        a = cc.cvae_function(self.x1, self.y1)
        a2 = cc.cvae_function(self.x2, self.y2)
        b = cc.cvae_function(self.x3, self.y3)
        
        if spec == 'a': # training process
            mu_q, logvar_q, z_q = a.latent_space_space(space_spec='train')
            mu_r1, logvar_r1, z_r = a.latent_space_space(space_spec='test')
            mu_r2, logvar_r2, x_samp = a.latent_space_space(space_spec='output')
            TotalLoss = cc.cvae_loss_function(mu_q, logvar_q, mu_r1, logvar_r1, self.x1, x_samp)
            rsq = cc.calculate_r2(x_samp, self.x1)
            
        if spec == 'a2': # validation process
            mu_q, logvar_q, z_q = a2.latent_space_space(space_spec='train')
            mu_r1, logvar_r1, z_r = a2.latent_space_space(space_spec='test')
            mu_r2, logvar_r2, x_samp = a2.latent_space_space(space_spec='output')
            TotalLoss = cc.cvae_loss_function(mu_q, logvar_q, mu_r1, logvar_r1, self.x2, x_samp)
            rsq = cc.calculate_r2(x_samp, self.x2)
            
        if spec == 'b': # testing process
            z_q = 0
            mu_r1, logvar_r1, z_r = b.latent_space_space(space_spec='test')
            mu_r2, logvar_r2, x_samp = b.latent_space_space(space_spec='output')
            TotalLoss = 0
            rsq = cc.calculate_r2(x_samp, self.x3)
        
        else:
            print('non existent spec!')
        
        return z_q, z_r, x_samp, TotalLoss, rsq
    

train_x, train_y = next(iter(train_loader))
val_x, val_y = next(iter(val_loader))
test_x, test_y = next(iter(test_loader))
network = ConditionalVariationalAutoEncoder(train_x, val_x, test_x, train_y, val_y, test_y)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

train_loss_history, train_r2_history, val_loss_history, val_r2_history = [], [], [], []

# Training & Validation
for epoch in range(num_epochs):
    train_total_loss = 0.0
    train_epoch_r2 = 0.0
    
    network.train()
    for inputs, targets in train_loader:
        targets = targets.view(-1, 3)
        
        z_q, z_r, x_samp, TotalLoss, rsq = network.doFunc(spec='a')
        
        optimizer.zero_grad()
        TotalLoss.backward()
        optimizer.step()
        
        train_total_loss += TotalLoss.item()
        train_epoch_r2 += rsq * inputs.size(0)
        
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Predicted shape: {x_samp.shape}")
    
    train_epoch_loss = train_total_loss / len(train_loader.dataset)
    train_epoch_r2 /= len(train_loader.dataset)
    
    train_loss_history.append(train_epoch_loss)
    train_r2_history.append(train_epoch_r2)
    
    val_total_loss = 0.0
    val_epoch_r2 = 0.0
    
    network.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            targets = targets.view(-1, 3)
            
            z_q, z_r, x_samp, ValTotalLoss, Valrsq = network.doFunc(spec='a2')
            
            val_total_loss += ValTotalLoss.item()
            val_epoch_r2 += Valrsq * inputs.size(0)
            
    print(f"Validation Input shape: {inputs.shape}")
    print(f"Validation Target shape: {targets.shape}")
    print(f"Validation Predicted shape: {x_samp.shape}")
    
    val_epoch_loss = val_total_loss / len(val_loader.dataset)
    val_epoch_r2 /= len(val_loader.dataset)
    
    val_loss_history.append(val_epoch_loss)
    val_r2_history.append(val_epoch_r2)

    print(f"\033[31;1mEpoch {epoch + 1}/{num_epochs}\033[m: Train Loss={train_epoch_loss}, Train R²={train_epoch_r2:.4f}, "
        f"Val Loss={val_epoch_loss}, Val R²={val_epoch_r2:.4f}")    

