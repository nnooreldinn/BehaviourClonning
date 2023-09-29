import torch
import torchvision
from torch.utils.data import (
    Dataset,
    DataLoader,)
import torch.utils.data
import numpy as np
import math
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CarDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label1 = torch.tensor(float(self.annotations.iloc[index, 3]))
        y_label2 = torch.tensor(float(self.annotations.iloc[index, 4]))
        #y_label1 = torch.cat((y_label1,y_label2),1)

        if self.transform:
            image = self.transform(image)

        return (image, y_label1,y_label2)
    

dataset = CarDataset(
    csv_file="driving_log.csv",
    root_dir="/home/nooreldin7/sim_udacity/IMG",
    transform=transforms.ToTensor())

train_set , test_set = torch.utils.data.random_split(dataset, [1400 , 500])

train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True,num_workers=2)
test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=True,num_workers=2)
dataiter = (iter(train_loader))
ccc = next(dataiter)
f,l ,c= ccc
print(next(iter(test_loader))[1].shape)
print(type(l))




class Model(nn.Module):


    def __init__(self):

        super(Model, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),

        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 13 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            
        )
        self.angles = nn.Linear(in_features=10,out_features=1)
        self.throttle = nn.Linear(in_features=10,out_features=1)

    def forward(self, input):
        """Forward pass."""
        input = input.view(input.size(0), 3, 160, 320)
        output = self.conv_layers(input)
        #print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        steering = self.angles(output)
        throttle = self.throttle(output)
        return steering,throttle





num_epochs = 3
batch_size = 10
learning_rate = 0.001
model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, angles,throttle) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        angles = angles.to(device)
        # Forward pass
        angles_out,throttle_out = model(images)
        steering_loss = criterion(angles_out, angles)
        throttle_loss = criterion(throttle_out,throttle)
        loss = steering_loss + throttle_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    val_steering_loss = 0.0
    val_throttle_loss = 0.0

    for i, (images, angles,throttle) in enumerate(test_loader):
        images = images.to(device)
        angles = angles.to(device)
        steering_outputs, throttle_outputs = model(images)

        val_steering_loss += criterion(steering_outputs, angles).item()
        val_throttle_loss += criterion(throttle_outputs, throttle).item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Val Steering Loss: {val_steering_loss/len(test_loader)}, Val Throttle Loss: {val_throttle_loss/len(test_loader)}")
