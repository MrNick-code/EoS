import data_man_pick as dmp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# paths: 
EOSXimgs_40 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_40b", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_40"]
EOSXimgs_50 = ["D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_50", "D:/Users/mathe/ML/EoS/IMG_DATA/EOSQ_low_50"]
# paths = EOSXimgs_40 e EOSXimgs_50
# shape = (50, 50, 4) e (60, 60, 4)

learning_rate = .001
batch_size = 64
num_epochs = 10

X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40 = dmp.get_data(paths=EOSXimgs_40, shape=(50, 50, 4))
# X_train50, X_val50, X_test50, Y_train50, Y_val50, Y_test50 = dmp.get_data(paths=EOSXimgs_50, shape=(60, 60, 4))

# pTensor= dataset.pytorch_tensor_samples([sample,sample1])
def NCHW(trainX, valX, testX, trainY, valY, testY):
    '''
    b = []
    for j in [trainY, valY, testY]:
        j = torch.from_numpy(j).float()  # Assuming "j" is a Numpy array
        b.append(j)


    a = []
    for i in [trainX, valX, testX]:
        i = torch.from_numpy(i).float()  # Assuming "i" is a Numpy array
        #i = i.torch.Tensor.permute(0, 3, 1, 2) # [batch size, number of channels, height, width]
        i = i.permute(0, 3, 1, 2) # [batch size, number of channels, height, width]
        a.append(i)
    # xt, xv, xtest = traindata.torch.Tensor.permute(0, 3, 1, 2), valdata.torch.Tensor.permute(0, 3, 1, 2), testdata.torch.Tensor.permute(0, 3, 1, 2)
    '''
    b = [torch.from_numpy(j).float() for j in [trainY, valY, testY]]
    a = [torch.from_numpy(i).float().permute(0, 3, 1, 2) for i in [trainX, valX, testX]]


    #train_dataset = TensorDataset(a[0], b[0])
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create TensorDataset and DataLoader for each dataset
    train_dataset = TensorDataset(a[0], b[0])
    val_dataset = TensorDataset(a[1], b[1])
    test_dataset = TensorDataset(a[2], b[2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

#a = NCHW(X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40)
#X_train40, X_val40, X_test40 = a[0], a[1], a[2]
train_loader40, val_loader40, test_loader40 = NCHW(X_train40, X_val40, X_test40, Y_train40, Y_val40, Y_test40)
# b = NCHW(X_train50, X_val50, X_test50)
# X_train50, X_val50, X_test50 = b[0], b[1], b[2]

def show_images(images, labels):
    """
    Display a set of images and their labels using matplotlib.
    The first column of `images` should contain the image indices,
    and the second column should contain the flattened image pixels
    reshaped into 28x28 arrays.
    """
    # Extract the image indices and reshaped pixels
    pixels = images.reshape(-1, 50, 50)

    # Create a figure with subplots for each image
    fig, axs = plt.subplots(
        ncols=len(images), nrows=1, figsize=(10, 3 * len(images))
    )

    # Loop over the images and display them with their labels
    for i in range(len(images)):
        # Display the image and its label
        axs[i].imshow(pixels[i], cmap="gray")
        axs[i].set_title("Label: {}".format(labels[i]))

        # Remove the tick marks and axis labels
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel("Index: {}".format(i))

    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.5)

    # Show the figure
    plt.show()


'''
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # nº of hidden units
        self.num_hidden = 8

        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8), padding='same'),
            torch.nn.Dropout(.2),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.PReLU(),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding='same'),
            torch.nn.Dropout(.2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.PReLU(),

            torch.nn.Flatten(),
            
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Dropout(.5),
            torch.nn.BatchNorm2d(num_features=64),

            torch.nn.Linear(in_features=64, out_features=3)
        )
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32 * 8 * 8),
            torch.nn.ReLU(),

            torch.nn.Unflatten(dim=1, unflattened_size=(32, 8, 8)),

            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(7, 7), padding='same'),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(8, 8), padding='same'),
            torch.nn.ReLU()
        )

    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder
        decoded = self.decoder(encoded)
        # Return both the encoded representation and the reconstructed output
        return encoded, decoded
    '''

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(3, 3), padding=1),
            torch.nn.Dropout(.2),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.PReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            torch.nn.Dropout(.2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.PReLU(),
            torch.nn.Flatten()
        )

        # Determinar a dimensão correta da saída da camada convolucional antes de Flatten
        dummy_input = torch.randn(1, 4, 50, 50)  # Um exemplo de entrada com as dimensões corretas
        flattened_size = self.encoder(dummy_input).view(-1).size(0)
        
        self.fc1 = torch.nn.Linear(in_features=flattened_size, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=3)
        
        self.decoder_fc1 = torch.nn.Linear(in_features=3, out_features=64)
        self.decoder_fc2 = torch.nn.Linear(in_features=64, out_features=flattened_size)

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = F.relu(self.fc1(encoded))
        encoded = self.fc2(encoded)
        decoded = F.relu(self.decoder_fc1(encoded))
        decoded = F.relu(self.decoder_fc2(decoded))
        decoded = decoded.view(-1, 32, 50, 50)  # Ajustar a dimensão para (batch_size, canais, altura, largura)
        decoded = self.decoder(decoded)
        return encoded, decoded


# Convert the training data to PyTorch tensors
# X_train = torch.from_numpy(X_train40)

# Create the autoencoder model and optimizer
model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define the loss function
criterion = torch.nn.MSELoss()

# Set the device to GPU if available, otherwise use CPU
# model.to(device)

# Create a DataLoader to handle batching of the training data
'''train_loader = torch.utils.data.DataLoader(
    X_train, batch_size=batch_size, shuffle=True
)'''



# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader40):
        # Get a batch of training data and move it to the device
        # data = data.to(device)

        # Forward pass
        encoded, decoded = model(inputs)

        # Compute the loss and perform backpropagation
        loss = criterion(decoded, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the running loss
        total_loss += loss.item() * inputs.size(0)

    # Print the epoch loss
    epoch_loss = total_loss / len(train_loader40.dataset)
    print(
        "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
    )


