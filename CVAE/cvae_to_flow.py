import torch
import torch.nn as nn
import CVAE_classes as cc
import psutil
from os import getpid
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
print(f"CPU memory in use: {psutil.Process(getpid()).memory_info().rss} bytes")

EOSXimgs_40 = ["C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSL_low_40b", "C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSQ_low_40"] # Path (50x50 imgs)
EOSXimgs_50 = ["C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSL_low_50", "C:\\Users\\mathe\\faculdade\\ML\\EoS\\IMG_DATA\\EOSQ_low_50"] # Path (60x60 imgs)

common_size, batch_size, num_epochs, learning_rate, normalization_term = (50, 50), 8, 30, 2e-5, 255

data_loader = cc.cvae_load(nt=normalization_term)

xy40, xy50 = data_loader.loadDataNorm(EOSXimgs_40, EOSXimgs_50)

train_loader, val_loader, test_loader = data_loader.NCHW(
                xy40[0], xy40[1], xy40[2], xy40[3], xy40[4], xy40[5],
                xy50[0], xy50[1], xy50[2], xy50[3], xy50[4], xy50[5],  
                cs=common_size, bs=batch_size)

class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, x1, x2, x3, y1, y2, y3):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.encoder2d = cc.cvae_function.conv2D_layers(self)
        self.encoderLin = cc.cvae_function.common_lin_layers(self)
    
    def forward(self, inputs): # must have torch function for later uses
        return self.doFunc(spec='forward', inputs=inputs)
    
    def doFunc(self, spec='', inputs=None):
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
        # print(self.y1)
        a = cc.cvae_function(self.x1, self.y1)
        a2 = cc.cvae_function(self.x2, self.y2)
        b = cc.cvae_function(self.x3, self.y3)
        
        # print(f"a.x e a.y sizes: {a.x.size()}, {a.y.size()}")
        
        if spec == 'a': # training process
            mu_q, logvar_q, z_q = a.latent_space_space(space_spec='train', zn=None)
            mu_r1, logvar_r1, z_r = a.latent_space_space(space_spec='test', zn=None)
            mu_r2, logvar_r2, x_samp = a.latent_space_space(space_spec='output', zn=z_q)
            TotalLoss = cc.cvae_loss_function(mu_q, logvar_q, mu_r1, logvar_r1, self.x1, x_samp)
            rsq = cc.calculate_r2(x_samp, self.x1)
            
        elif spec == 'a2': # validation process
            mu_q, logvar_q, z_q = a2.latent_space_space(space_spec='train', zn=None)
            mu_r1, logvar_r1, z_r = a2.latent_space_space(space_spec='test', zn=None)
            mu_r2, logvar_r2, x_samp = a2.latent_space_space(space_spec='output', zn=z_q)
            TotalLoss = cc.cvae_loss_function(mu_q, logvar_q, mu_r1, logvar_r1, self.x2, x_samp)
            rsq = cc.calculate_r2(x_samp, self.x2)
            
        elif spec == 'b': # testing process
            z_q = 0
            mu_r1, logvar_r1, z_r = b.latent_space_space(space_spec='test', zn=None)
            mu_r2, logvar_r2, x_samp = b.latent_space_space(space_spec='output', zn=z_r)
            TotalLoss = 0
            rsq = cc.calculate_r2(x_samp, self.x3)
        
        elif spec == 'forward':  # forward pass case
            encoded_2d_output = self.encoder2d(inputs)
            self.encoderLin = nn.Linear(76832, 3)
            
            #print(f'encoded_2d_output shape: {encoded_2d_output.shape}')
            flattened_output = encoded_2d_output.view(encoded_2d_output.size(0), -1)  # Flatten: [8, 32*49*49]
            #print(f'flattened_output shape: {flattened_output.shape}')
            lin_output = self.encoderLin(flattened_output)

            mu, logvar, decoded, predicted_values = None, None, flattened_output, lin_output
            return mu, logvar, decoded, predicted_values
       
        else:
            print(f'non existent spec value: {spec}')
            return None, None, None, None, None
        
        return z_q, z_r, x_samp, TotalLoss, rsq


train_y, train_x = next(iter(train_loader))
val_y, val_x = next(iter(val_loader))
test_y, test_x = next(iter(test_loader))
print(train_y.shape)
print(len(train_y))
print(train_x)
network = ConditionalVariationalAutoEncoder(train_x, val_x, test_x, train_y, val_y, test_y)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

train_loss_history, train_r2_history, val_loss_history, val_r2_history = [], [], [], []

print(f"train_x size: {train_x.size()}")
print(f"train_y size: {train_y.size()}")

# Training & Validation
print('\033[31;1m Training & Validation Phase\033[m')
for epoch in range(num_epochs):
    train_total_loss = 0.0
    train_epoch_r2 = 0.0
    
    network.train()
    for inputs, targets in train_loader:
        #print(f"inputs size (x): {inputs.size()}")
        #print(f"targets size (y): {targets.size()}") 
        targets = targets.view(-1, 3)
        
        z_q, z_r, x_samp, TotalLoss, rsq = network.doFunc(spec='a')
        
        #print(f"z_q size: {z_q.size()}")
        #print(f"z_r size: {z_r.size()}")
        #print(f"x_samp size (output): {x_samp.size()}")
        
        #print(f"TotalLoss: {TotalLoss}")
        
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

# Data Analysis
# Original vs Reconstructed image function
def show_images(inputs, outputs, n=5):
    inputs = inputs[:n]
    outputs = outputs[:n]

    # Se os tensores forem 4D, apenas permute para a forma correta para exibir
    if len(inputs.shape) == 4:  # Se for um tensor 4D (como imagens)
        inputs = inputs.permute(0, 2, 3, 1).detach().cpu()
    if len(outputs.shape) == 4:  # Se for um tensor 4D (como imagens)
        outputs = outputs.permute(0, 2, 3, 1).detach().cpu()

    # Se os tensores forem 1D (por exemplo, valores achatados), reshape para 2D
    if len(inputs.shape) == 2:  # Se for 2D (após ser achatado)
        # Ajuste o reshape para corresponder à dimensão da imagem esperada (como 49x49)
        inputs = inputs.view(-1, 49, 49)  # Ou as dimensões originais da imagem
    if len(outputs.shape) == 2:  # Se for 2D (após ser achatado)
        outputs = outputs.view(-1, 49, 49)  # Ajuste conforme necessário

    fig, axs = plt.subplots(n, 2, figsize=(10, 15))
    for i in range(n):
        axs[i, 0].imshow(inputs[i])  # Visualize a imagem original
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Input Image')
        axs[i, 1].imshow(outputs[i])  # Visualize a imagem reconstruída
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Reconstructed Image')

    plt.show()

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

### Results Analysis
network.eval()
predictions, real_values = [], []
first_iteration = True

with torch.no_grad():
    for inputs, targets in test_loader:
        mu, logvar, decoded, predicted_values = network(inputs)  # ConvVAE output

        predictions.append(predicted_values)  # Y^
        real_values.append(targets)  # Y

        if first_iteration:
            show_images(inputs, decoded)
            first_iteration = False


# Single arrays
predictions = torch.cat(predictions, dim=0).cpu().numpy()
real_values = torch.cat(real_values, dim=0).cpu().numpy()

print(f'real_value shape: {real_values.shape}')
print(f'prediction shape: {predictions.shape}')

print(f'\033[32;1mV2² test R²: {r2_score(real_values[0], predicted_values[0])}\033[m')
print(f'\033[32;1mV3² test R²: {r2_score(real_values[1], predicted_values[1])}\033[m')
print(f'\033[32;1mV4² test R²: {r2_score(real_values[2], predicted_values[2])}\033[m')

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
