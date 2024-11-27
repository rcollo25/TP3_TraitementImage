from torch import flatten
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, in_channel, output_dim):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, 3)
        self.relu1 = nn.ReLU()
        self.BN1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, 2)
        self.relu2 = nn.ReLU()
        self.BN2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.relu3 = nn.ReLU()
        self.BN3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, 2)
        self.relu4 = nn.ReLU()
        self.BN4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.relu5 = nn.ReLU()
        self.BN5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3, 2)
        self.relu6 = nn.ReLU()
        self.BN6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(186624, 120)
        self.relufc1 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relufc2 = nn.ReLU()

        self.fc3 = nn.Linear(84, output_dim)
        self.act_output = nn.Softmax(dim=1)


    def forward(self, x):

        z1 = self.conv1(x)
        a1 = self.relu1(z1)
        a1 = self.BN1(a1)

        z2 = self.conv2(a1)
        a2 = self.relu2(z2)
        a2 = self.BN2(a2)

        z3 = self.conv3(a2)
        a3 = self.relu3(z3)
        a3 = self.BN3(a3)

        z4 = self.conv4(a3)
        a4 = self.relu4(z4)
        a4 = self.BN4(a4)

        z5 = self.conv5(a4)
        a5 = self.relu5(z5)
        a5 = self.BN5(a5)

        z6 = self.conv6(a5)
        a6 = self.relu6(z6)
        a6 = self.BN6(a6)

        a8 = flatten(a6, 1)

        z9 = self.fc1(a8)
        a9 = self.relufc1(z9)

        z10 = self.fc2(a9)
        a10 = self.relufc2(z10)

        z11 = self.fc3(a10)
        y = self.act_output(z11)

        return y

class CNNClassifier_1(nn.Module):
    def __init__(self, in_channel, output_dim):
        super(CNNClassifier_1, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, 3)
        self.relu1 = nn.ReLU()
        self.BN1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, 2)
        self.relu2 = nn.ReLU()
        self.BN2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.relu3 = nn.ReLU()
        self.BN3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, 2)
        self.relu4 = nn.ReLU()
        self.BN4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.relu5 = nn.ReLU()
        self.BN5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3, 2)
        self.relu6 = nn.ReLU()
        self.BN6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3)
        self.relu7 = nn.ReLU()
        self.BN7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, 3, 2)
        self.relu8 = nn.ReLU()
        self.BN8 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(73728, 120)
        self.relufc1 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relufc2 = nn.ReLU()

        self.fc3 = nn.Linear(84, output_dim)
        self.act_output = nn.Softmax(dim=1)


    def forward(self, x):

        z1 = self.conv1(x)
        a1 = self.relu1(z1)
        a1 = self.BN1(a1)

        z2 = self.conv2(a1)
        a2 = self.relu2(z2)
        a2 = self.BN2(a2)

        z3 = self.conv3(a2)
        a3 = self.relu3(z3)
        a3 = self.BN3(a3)

        z4 = self.conv4(a3)
        a4 = self.relu4(z4)
        a4 = self.BN4(a4)

        z5 = self.conv5(a4)
        a5 = self.relu5(z5)
        a5 = self.BN5(a5)

        z6 = self.conv6(a5)
        a6 = self.relu6(z6)
        a6 = self.BN6(a6)

        z7 = self.conv7(a6)
        a7 = self.relu7(z7)
        a7 = self.BN7(a7)

        z8 = self.conv8(a7)
        a8 = self.relu8(z8)
        a8 = self.BN8(a8)

        a8 = flatten(a8, 1)

        z9 = self.fc1(a8)
        a9 = self.relufc1(z9)

        z10 = self.fc2(a9)
        a10 = self.relufc2(z10)

        z11 = self.fc3(a10)
        y = self.act_output(z11)

        return y


class CNNClassifier_2(nn.Module):
    def __init__(self, in_channel, output_dim):
        super(CNNClassifier_2, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, 3)
        self.relu1 = nn.ReLU()
        self.BN1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, 2)
        self.relu2 = nn.ReLU()
        self.BN2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.relu3 = nn.ReLU()
        self.BN3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, 2)
        self.relu4 = nn.ReLU()
        self.BN4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.relu5 = nn.ReLU()
        self.BN5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3, 2)
        self.relu6 = nn.ReLU()
        self.BN6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3)
        self.relu7 = nn.ReLU()
        self.BN7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, 3, 2)
        self.relu8 = nn.ReLU()
        self.BN8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 1024, 3)
        self.relu9 = nn.ReLU()
        self.BN9 = nn.BatchNorm2d(1024)

        self.conv10 = nn.Conv2d(1024, 1024, 3, 2)
        self.relu10 = nn.ReLU()
        self.BN10 = nn.BatchNorm2d(1024)

        self.conv11 = nn.Conv2d(1024, 2048, 3)
        self.relu11 = nn.ReLU()
        self.BN11 = nn.BatchNorm2d(2048)

        self.conv12 = nn.Conv2d(2048, 2048, 3, 2)
        self.relu12 = nn.ReLU()
        self.BN12 = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(32768, 120)
        self.relufc1 = nn.ReLU()

        self.fc2 = nn.Linear(120, 84)
        self.relufc2 = nn.ReLU()

        self.fc3 = nn.Linear(84, output_dim)
        self.act_output = nn.Softmax(dim=1)


    def forward(self, x):

        z1 = self.conv1(x)
        a1 = self.relu1(z1)
        a1 = self.BN1(a1)

        z2 = self.conv2(a1)
        a2 = self.relu2(z2)
        a2 = self.BN2(a2)

        z3 = self.conv3(a2)
        a3 = self.relu3(z3)
        a3 = self.BN3(a3)

        z4 = self.conv4(a3)
        a4 = self.relu4(z4)
        a4 = self.BN4(a4)

        z5 = self.conv5(a4)
        a5 = self.relu5(z5)
        a5 = self.BN5(a5)

        z6 = self.conv6(a5)
        a6 = self.relu6(z6)
        a6 = self.BN6(a6)

        z7 = self.conv7(a6)
        a7 = self.relu7(z7)
        a7 = self.BN7(a7)

        z8 = self.conv8(a7)
        a8 = self.relu8(z8)
        a8 = self.BN8(a8)

        z9 = self.conv9(a8)
        a9 = self.relu9(z9)
        a9 = self.BN9(a9)

        z10 = self.conv10(a9)
        a10 = self.relu10(z10)
        a10 = self.BN10(a10)

        z11 = self.conv11(a10)
        a11 = self.relu11(z11)
        a11 = self.BN11(a11)

        z12 = self.conv12(a11)
        a12 = self.relu12(z12)
        a12 = self.BN12(a12)

        a12 = flatten(a12, 1)

        z13 = self.fc1(a12)
        a13 = self.relufc1(z13)

        z14 = self.fc2(a13)
        a14 = self.relufc2(z14)

        z15 = self.fc3(a14)
        y = self.act_output(z15)

        return y


class CNNClassifier_3(nn.Module):
    def __init__(self, in_channel, output_dim):
        super(CNNClassifier_3, self).__init__()

        # Couches convolutionnelles
        self.conv1 = nn.Conv2d(in_channel, 64, 3, padding=1)  # 64 filtres de taille 3x3
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 128 filtres, stride 2
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)  # 256 filtres
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)  # 512 filtres, stride 2
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(512)

        # Global Average Pooling (GAP) pour éviter trop de paramètres dans les couches fully connected
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Couche fully connected après le GAP
        self.fc1 = nn.Linear(512, 512)  # Augmenter la taille de cette couche
        self.relu_fc1 = nn.ReLU()

        # Augmentation de la taille des couches fully connected pour mieux capturer des relations complexes
        self.fc2 = nn.Linear(512, 256)
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(256, output_dim)  # Sortie à 20 classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)

        # Appliquer le Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Aplatir pour passer à la couche fully connected

        x = self.fc1(x)
        x = self.relu_fc1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)

        x = self.fc3(x)
        x = self.softmax(x)  # Calcul des probabilités pour chaque classe

        return x
