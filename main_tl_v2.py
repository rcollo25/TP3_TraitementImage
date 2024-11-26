import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import random_split

from CNN_Architectures import CNNClassifier_1, CNNClassifier_2
from other_tools import get_model_information
from train_process import train_model
from test_process import test_model

########################################################################################################################
#                                                    USER PARAMETERS                                                   #
########################################################################################################################

# Define the path of the dataset to use
dataset_path = ("C:\\Users\\ryan4\\OneDrive\\Documents\\RYAN\\ISEN\\Année24-25_UQAC\\Trimestre1\\Traitement_image\\"
                "pythonProject\\TP3_TraitementImage\\Dataset 1")

# Define the path where to save the results
results_path = ("C:\\Users\\ryan4\\OneDrive\\Documents\\RYAN\\ISEN\\Année24-25_UQAC\\Trimestre1\\Traitement_image\\"
                "pythonProject\\TP3_TraitementImage\\Results")

# Define the number of epochs of the model training
epoch_number = 10

# Define the size of the mini-batch
batch_size = 32

# Define the learning rate
learning_rate = 0.001

# Define the methode to use
choice_methode = 1  # 0 = transfer learning ; 1 = CNN

########################################################################################################################
#                                       CREATE A FOLDER AND FILE TO SAVE RESULTS                                       #
########################################################################################################################

# Get the date and time
now = datetime.now()
# Create the folder name
my_folder_name = now.strftime("%Y-%m-%d_%H" + "h" + "%M" + "min" + "%S" + "sec")
# Create the folder
os.makedirs(os.path.join(results_path, my_folder_name))
# Print a message in the console
print("\nResult folder created")

# Create and open a txt file to store information about the model performances
txt_file = open(os.path.join(results_path, my_folder_name, "Results.txt"), "a")

# Write information about the architecture used for classification
txt_file.write("Model Information\n")
txt_file.write("    - Model: ResNet50\n")
txt_file.write("    - Task: Classification\n")
txt_file.write("    - Type of training: Transfer learning\n\n")
# Write information about hyperparameters
txt_file.write("Hyperparameters\n")
txt_file.write("    - Epoch number: " + str(epoch_number) + "\n")
txt_file.write("    - Batch size: " + str(batch_size) + "\n")
txt_file.write("    - Learning rate: " + str(learning_rate) + "\n\n")

########################################################################################################################
#                                                   LOAD THE DATASET                                                   #
########################################################################################################################

""" Data Transformation """
if choice_methode == 0:
    # Define the transformations to apply to x data (follow ones applied for ResNet50_Weights.IMAGENET1K_V2)
    transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
else:
    # Define the transformations to apply to x data (for CNN)
    transform = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])

# Define the transformations to apply to y labels
target_transform = \
    transforms.Compose([transforms.Lambda(lambda y: torch.zeros(7, dtype=torch.float).scatter_(0, torch.tensor(y),
                                                                                               value=1))])

"""" Train, validation and test sets """
# Load the dataset by applying transformations
dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform, target_transform=target_transform)

# Divide the dataset into a train, validation and test sets
generator1 = torch.Generator().manual_seed(42)
train_set, validation_set, test_set = random_split(dataset,[0.7, 0.15, 0.15], generator=generator1)

# Create the Python iterator for the train set (creating mini-batches)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# Create the Python iterator for the validation set
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
# Create the Python iterator for the test set
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Load the first batch of images from the train set
images, labels = next(iter(train_loader))

# Get the shape of the images
image_shape = list(images.data.shape)
# Get automatically the number of channels of images
image_channel = image_shape[1]

# Get the number of classes from the dataset
classes = dataset.classes
class_number = len(list(classes))

########################################################################################################################
#                                  CHECK GPU AVAILABILITY AND CREATE THE NETWORK MODEL                                 #
########################################################################################################################

# Check if GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if choice_methode == 0:
    # Instantiate the model to exploit
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Replace the last fully connected layer to fit with the number of classes of the dataset
    model.fc = nn.Linear(model.fc.in_features, class_number)

    # Place the model into the GPU if available
    model = model.to(device)
else:
    # Instantiate and move the model to GPU
    model = CNNClassifier_1(in_channel=image_channel, output_dim=class_number).to(device)

# Print information about the model
get_model_information(model, txt_file)

########################################################################################################################
#                                          SET THE LOSS FUNCTION AND OPTIMIZER                                         #
########################################################################################################################

# Create the loss function
loss_function = nn.CrossEntropyLoss()
# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

########################################################################################################################
#                                           TRAIN AND TEST THE NETWORK MODEL                                           #
########################################################################################################################

# Train the neural network
model = train_model(epoch_number, train_loader, validation_loader, model, optimizer, loss_function, device,
                    results_path, my_folder_name)
# Test the neural network
test_model(test_loader, model, loss_function, device, classes, txt_file)

# Close your txt file
txt_file.close()

########################################################################################################################
#                                                SAVE THE TRAINED MODEL                                                #
########################################################################################################################

if choice_methode == 0:
    print("\nThe program are saving the trained model. Please wait ...")
    torch.save(model.state_dict(), os.path.join(results_path, my_folder_name, "my_model.pth"))
    print("\nModel saved")
