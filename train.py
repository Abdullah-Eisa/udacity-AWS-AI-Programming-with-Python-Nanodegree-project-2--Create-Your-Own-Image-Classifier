
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import json
from PIL import Image
import utils
import model_functions
import argparse

arch = {"vgg16":25088,"densenet121":1024}


# create a parser for train.py argument
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--gpu', action="store", default="gpu")

# python train.py data_dir ./flowers/ --save_dir ./checkpoint.pth --arch vgg16 --learning_rate 0.001 --hidden_units 4096 --epochs 3 --dropout 0.2 --gpu gpu 



args = parser.parse_args()

data_dir = args.data_dir
saving_path = args.save_dir
struct = args.arch
hidden_units = args.hidden_units
lr = args.learning_rate
power = args.gpu
epochs = args.epochs
dropout = args.dropout

device = torch.device("cuda" if torch.cuda.is_available() and power== 'gpu' else "cpu")


def main():
    trainloader, validloader, testloader, train_data = utils.load_data(data_dir)
    model, criterion = model_functions.nn_prepare(struct,dropout,hidden_units,lr,power)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
    # Train Model
    steps = 0
    running_loss = 0
    print_every = 5
    print("--Training starting--")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
          
            if torch.cuda.is_available() and power =='gpu':
                inputs, labels = inputs.to(device), labels.to(device)
                model = model.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    
    # saving model parameters
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :struct,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                saving_path)
    print("checkpoint Saved")
    
if __name__ == "__main__":
    main()




