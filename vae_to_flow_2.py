#  Convolutional Variational Autoencoder

import data_man_pick as dmp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import r2_score

# paths:
# paths = EOSXimgs_40 e EOSXimgs_50
# shape = (50, 50, 4) e (60, 60, 4)
EOSXimgs_40 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_40b", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_40"]
# EOSXimgs_50 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_50", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_50"]

learning_rate = 1e-4
batch_size = 64
num_epochs = 20

# load data
X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40 = dmp.get_data(paths=EOSXimgs_40, shape=(50, 50, 4))
# X_train50, X_val50, X_test50, Y_train50, Y_val50, Y_test50 = dmp.get_data(paths=EOSXimgs_50, shape=(60, 60, 4))

# data norm.j
def normalize(train, val, test):
    return train / 255, val / 255, test / 255 

X_train40, X_val40, X_test40 = normalize(X_train40, X_val40, X_test40)


# tensorflow --> pytorch
def NCHW(trainX, valX, testX, trainY, valY, testY):
    b = [torch.from_numpy(j).float() for j in [trainY, valY, testY]]
    a = [torch.from_numpy(i).float().permute(0, 3, 1, 2) for i in [trainX, valX, testX]]

    train_dataset = TensorDataset(a[0], b[0])
    val_dataset = TensorDataset(a[1], b[1])
    test_dataset = TensorDataset(a[2], b[2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

train_loader40, val_loader40, test_loader40 = NCHW(X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40)
# train_loader50, val_loader50, test_loader50 = NCHW(X_train50, X_val50, X_test50, Y_train50, Y_val50, Y_test50)

# VAE class
class ConvVariationalAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8), padding=3),
            torch.nn.Dropout(.2),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.PReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding=3),
            torch.nn.Dropout(.2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.PReLU()
        )

        # correctly output dimension of conv layer before flatten
        dummy_input = torch.randn(1, 4, 50, 50)
        conv_out = self.encoder(dummy_input)
        self.flattened_size = conv_out.view(-1).size(0)
        self.conv_output_shape = conv_out.size()[1:] # all dimensions but batch size

        # latent space
        self.fc_mu = torch.nn.Linear(in_features=self.flattened_size, out_features=3)
        self.fc_logvar = torch.nn.Linear(in_features=self.flattened_size, out_features=3)

        # Decoder
        self.decoder_fc1 = torch.nn.Linear(in_features=3, out_features=64)
        self.decoder_fc2 = torch.nn.Linear(in_features=64, out_features=self.flattened_size)

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(7, 7), padding=3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(8, 8), padding=3),
            torch.nn.ReLU()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        conv_encoded = self.encoder(x)
        encoded = conv_encoded.view(conv_encoded.size(0), -1)

        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        z = self.reparameterize(mu, logvar)

        decoded = F.relu(self.decoder_fc1(z))
        decoded = F.relu(self.decoder_fc2(decoded))
        batch_size = decoded.size(0)
        decoded = decoded.view(batch_size, *self.conv_output_shape)
        decoded = self.decoder(decoded)

        return mu, logvar, decoded

# instace of model and optimizer
vae_model = ConvVariationalAutoEncoder()
optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

# define the loss function
def vae_loss_function(decoded, inputs, mu, logvar, beta=0.5):
    recon_loss = F.mse_loss(decoded, inputs, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_divergence / inputs.size(0)

# define the metric r²
def calculate_r2(inputs, outputs):
    inputs_np = inputs.view(-1).detach().cpu().numpy()
    outputs_np = outputs.view(-1).detach().cpu().numpy()
    return r2_score(inputs_np, outputs_np)

# init lists for the history of loss and r²
train_loss_history = []
train_r2_history = []
val_loss_history = []
val_r2_history = []

# training loop
for epoch in range(num_epochs):
    train_total_loss = 0.0
    train_epoch_r2 = 0.0
    
    # training phase
    vae_model.train()
    for inputs, targets in train_loader40:
        # forward pass
        mu, logvar, decoded = vae_model(inputs)

        loss = vae_loss_function(decoded, inputs, mu, logvar)
        r2 = calculate_r2(inputs, decoded)

        # perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the running loss & r²
        train_total_loss += loss.item()
        train_epoch_r2 += r2 * inputs.size(0)

    # calculate average training loss & r² for the epoch
    train_epoch_loss = train_total_loss / len(train_loader40.dataset)
    train_epoch_r2 /= len(train_loader40.dataset)
    
    # store training loss & r²
    train_loss_history.append(train_epoch_loss)
    train_r2_history.append(train_epoch_r2)

    val_total_loss = 0.0
    val_epoch_r2 = 0.0
    
    # validation phase
    vae_model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader40:
            # forward pass
            mu, logvar, decoded = vae_model(inputs)

            val_loss = vae_loss_function(decoded, inputs, mu, logvar)
            val_r2 = calculate_r2(inputs, decoded)

            # update the running loss & r²
            val_total_loss += val_loss.item()
            val_epoch_r2 += val_r2 * inputs.size(0)

    # calculate average val loss & r² for the epoch
    val_epoch_loss = val_total_loss / len(val_loader40.dataset)
    val_epoch_r2 /= len(val_loader40.dataset)
    
    # store validation loss & r²
    val_loss_history.append(val_epoch_loss)
    val_r2_history.append(val_epoch_r2)

    print("Epoch {}/{}: Train Loss={:.4f}, Train R²={:.4f}, Val Loss={:.4f}, Val R²={:.4f}".format(
        epoch + 1, num_epochs, train_epoch_loss, train_epoch_r2, val_epoch_loss, val_epoch_r2))

# data visualization
def show_images(inputs, outputs, n=5):
    inputs = inputs[:n]
    outputs = outputs[:n]

    # permute: general shape N, H, W, C
    # detach: tensor out of gradient operations (freeze values)
    # cpu: for tensor --> array, it must be at CPU
    # numpy: tensor --> array
    inputs = inputs.permute(0, 2, 3, 1).detach().cpu().numpy()
    outputs = outputs.permute(0, 2, 3, 1).detach().cpu().numpy()

    # garantee that the imgs has pixel range at [0, 1]
    inputs = np.clip(inputs, 0, 1)
    outputs = np.clip(outputs, 0, 1)

    fig, axes = plt.subplots(n, 2)
    for i in range(n):
        axes[i, 0].imshow(inputs[i])
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(outputs[i])
        axes[i, 1].set_title("Output")
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

# ---------------------------------------------------------------------------------------
### Results Analysis
vae_model.eval()

# Armazenar previsões e valores reais para análise
predictions, real_values = [], []

with torch.no_grad():
    for inputs, targets in test_loader40:
        mu, logvar, decoded = vae_model(inputs)  # Saídas do VAE

        # Aqui você deve definir como obter as previsões corretamente
        predicted_values = decoded  # No seu caso, decoded já representa a saída que você deseja prever

        predictions.append(predicted_values)  # Append das previsões (Y^)
        real_values.append(targets)  # Append dos valores reais (Y)

        show_images(inputs, decoded)  # Mostrar imagens para referência, se necessário
        break  # Remova o 'break' se quiser iterar por todo o conjunto de testes

# Concatenar todas as previsões e valores reais em arrays únicos
predictions = torch.cat(predictions, dim=0).cpu().numpy()
real_values = torch.cat(real_values, dim=0).cpu().numpy()

print(f'real_val: {real_values.shape}')
print(f'predic: {predictions.shape}')

# Pegando as 3 primeiras dimensões para o scatter plot
predictions = predictions[:, :3]
real_values = real_values[:, :3]

fig, axs = plt.subplots(1, 3, figsize=(12, 5))

for i in range(3):
    axs[i].scatter(real_values[:, i], predictions[:, i], c='seagreen', label=f'v_{i+2}²')
    axs[i].plot(real_values[:, i], real_values[:, i], color='lime', linestyle='--')  # linha de referência
    axs[i].set_xlabel(f'Real v_{i+2}²')
    axs[i].set_ylabel(f'Predicted v_{i+2}²')
    axs[i].legend()

plt.suptitle('Real vs. Predicted Values for Each Output Dimension')
plt.show()
