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

# create a parser for train.py argument
parser = argparse.ArgumentParser()

parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)


args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
json_name = args.category_names
path = args.checkpoint

device = torch.device("cuda" if torch.cuda.is_available() and args.gpu == 'gpu' else "cpu")

def main():
    model=fmodel.load_checkpoint(path)
    with open(json_name, 'r') as json_file:
        name = json.load(json_file)
        
    probabilities = model_functions.predict(path_image, model, number_of_outputs, device)
    probability = np.array(probabilities[0][0])
    labels = [name[str(index + 1)] for index in np.array(probabilities[1][0])]
    
    
    for i in range(number_of_outputs):
        print("{} with a probability of {}".format(labels[i], probability[i]))
    
    print("Prediction Finished !")

    
if __name__== "__main__":
    main()
