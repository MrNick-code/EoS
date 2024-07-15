import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import data_man_pick as dmp
import matplotlib.pyplot as plt

EOSXimgs_40 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_40b", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_40"] # Path (50x50 imgs)

learning_rate, batch_size, num_epochs = 1e-5, 64, 50 # Training (initial: 1e-4, 64)
patience, min_delta = 5, 1e-4 # Callback
alpha, betta, gamma = 1, .5, 5 # Loss proportion (initial: 1, .5, 5)

X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40 = dmp.get_data(paths=EOSXimgs_40, shape=(50, 50, 4)) # Load data

# Normalization function
def normalize(train, val, test):
    return train / 255, val / 255, test / 255

X_train40, X_val40, X_test40 = normalize(X_train40, X_val40, X_test40) # Normalize data

# [Num samples, Hidht, Weight, Chanells] --> [Num samples, Chanells, Hidht, Weight]
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

train_loader40, val_loader40, test_loader40 = NCHW(X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40) # Transform data

# Convolutional VAE
class ConvVariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8), padding=3),
            nn.Dropout(.2),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding=3),
            nn.Dropout(.2),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )

        # Encoder out dimension
        dummy_input = torch.randn(1, 4, 50, 50)
        conv_out = self.encoder(dummy_input)
        self.flattened_size = conv_out.view(-1).size(0)
        self.conv_output_shape = conv_out.size()[1:]

        # Latent space
        self.fc_mu = nn.Linear(in_features=self.flattened_size, out_features=3)
        self.fc_logvar = nn.Linear(in_features=self.flattened_size, out_features=3)

        # Decoder
        self.decoder_fc1 = nn.Linear(in_features=3, out_features=64)
        self.decoder_fc2 = nn.Linear(in_features=64, out_features=self.flattened_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(7, 7), padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(8, 8), padding=3),
            nn.ReLU()
        )

        self.fc_pred = nn.Linear(in_features=3, out_features=3)

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

        predicted_values = self.fc_pred(z)

        decoded = F.relu(self.decoder_fc1(z))
        decoded = F.relu(self.decoder_fc2(decoded))
        batch_size = decoded.size(0)
        decoded = decoded.view(batch_size, *self.conv_output_shape)
        decoded = self.decoder(decoded)

        return mu, logvar, decoded, predicted_values

vae_model = ConvVariationalAutoEncoder()
optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

# Loss function
def vae_loss_function(decoded, inputs, mu, logvar, predicted_values, targets, a=alpha, b=betta, c=gamma):
    recon_loss = F.mse_loss(decoded, inputs, reduction='mean')
    kl_divergence = ( -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) ) / inputs.size(0)
    prediction_loss = F.mse_loss(predicted_values, targets, reduction='mean')
    return a*recon_loss + b*kl_divergence + c*prediction_loss

# R² metric function
def calculate_r2(inputs, outputs):
    inputs_np = inputs.view(-1).detach().cpu().numpy()
    outputs_np = outputs.view(-1).detach().cpu().numpy()
    return r2_score(inputs_np, outputs_np)

# Callback function
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

train_loss_history = []
train_r2_history = []
val_loss_history = []
val_r2_history = []
early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

# Training & Validation
for epoch in range(num_epochs):
    train_total_loss = 0.0
    train_epoch_r2 = 0.0
    
    vae_model.train()
    for inputs, targets in train_loader40:
        targets = targets.view(-1, 3)  # targets must have the sasme shape as predicted_values!

        mu, logvar, decoded, predicted_values = vae_model(inputs)

        loss = vae_loss_function(decoded, inputs, mu, logvar, predicted_values, targets)
        r2 = calculate_r2(targets, predicted_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
        train_epoch_r2 += r2 * inputs.size(0)

    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Predicted shape: {predicted_values.shape}")

    train_epoch_loss = train_total_loss / len(train_loader40.dataset)
    train_epoch_r2 /= len(train_loader40.dataset)
    
    train_loss_history.append(train_epoch_loss)
    train_r2_history.append(train_epoch_r2)

    val_total_loss = 0.0
    val_epoch_r2 = 0.0
    
    vae_model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader40:
            targets = targets.view(-1, 3)  # targets must have the same shape as predicted_values again

            mu, logvar, decoded, predicted_values = vae_model(inputs)

            val_loss = vae_loss_function(decoded, inputs, mu, logvar, predicted_values, targets)
            val_r2 = calculate_r2(targets, predicted_values)

            val_total_loss += val_loss.item()
            val_epoch_r2 += val_r2 * inputs.size(0)

    print(f"Validation Input shape: {inputs.shape}")
    print(f"Validation Target shape: {targets.shape}")
    print(f"Validation Predicted shape: {predicted_values.shape}")

    val_epoch_loss = val_total_loss / len(val_loader40.dataset)
    val_epoch_r2 /= len(val_loader40.dataset)
    
    val_loss_history.append(val_epoch_loss)
    val_r2_history.append(val_epoch_r2)

    print("Epoch {}/{}: Train Loss={:.4f}, Train R²={:.4f}, Val Loss={:.4f}, Val R²={:.4f}".format(
        epoch + 1, num_epochs, train_epoch_loss, train_epoch_r2, val_epoch_loss, val_epoch_r2))
    
    '''early_stopping(val_epoch_loss)
    if early_stopping.early_stop:
        break'''

# Original vs Reconstructed image function
def show_images(inputs, outputs, n=5):
    inputs = inputs[:n]
    outputs = outputs[:n]

    inputs = inputs.permute(0, 2, 3, 1).detach().cpu()
    outputs = outputs.permute(0, 2, 3, 1).detach().cpu()

    fig, axs = plt.subplots(n, 2, figsize=(10, 15))
    for i in range(n):
        axs[i, 0].imshow(inputs[i])
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Input Image')
        axs[i, 1].imshow(outputs[i])
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

# ---------------------------------------------------------------------------------------
### Results Analysis
vae_model.eval()
predictions, real_values = [], []

with torch.no_grad():
    for inputs, targets in test_loader40:
        mu, logvar, decoded, predicted_values = vae_model(inputs)  # ConvVAE output

        predictions.append(predicted_values)  # Y^
        real_values.append(targets)  # Y

        show_images(inputs, decoded)
        break  # Remova o 'break' se quiser iterar por todo o conjunto de testes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Single arrays
predictions = torch.cat(predictions, dim=0).cpu().numpy()
real_values = torch.cat(real_values, dim=0).cpu().numpy()

print(f'real_val: {real_values.shape}')
print(f'predic: {predictions.shape}')

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
