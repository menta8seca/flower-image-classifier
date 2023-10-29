import argparse
from torchvision import datasets,transforms
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

def get_train_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dir', type=str, help='path to the folder of flower images')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'checkpoint save path')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'CNN Model Architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.0003, help = 'optimizer\'s learning rate')
    parser.add_argument('--hidden_units', type = int, default = 4096, help = 'hidden layer units')
    parser.add_argument('--epochs', type = int, default = 32, help = 'epochs number')
    parser.add_argument('--gpu', action='store_true', default=False, help='Enable GPU')
    
    return parser.parse_args()

def prepaire_data(dir):
    train_dir = dir + '/train'
    valid_dir = dir + '/valid'
    test_dir = dir + '/test'

    training_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    testing_transforms = transforms.Compose([transforms.Resize(225),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    training_datasets =datasets.ImageFolder(train_dir,transform=training_transforms)
    valid_datasets =datasets.ImageFolder(valid_dir,transform=testing_transforms)
    testing_datasets =datasets.ImageFolder(test_dir,transform=testing_transforms)

    trainloader = torch.utils.data.DataLoader(training_datasets,batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets,batch_size=64)
    testloader = torch.utils.data.DataLoader(testing_datasets,batch_size=64)
    
    return trainloader,validloader,testloader,training_datasets.class_to_idx

def get_predict_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image', type=str, help='path to image')
    parser.add_argument('checkpoint', type=str, help='path to model checkpoint')
    parser.add_argument('--top_k', type = int, default = 5, help = 'number of most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'mapping of category names')
    parser.add_argument('--gpu', action='store_true', default=False, help='Enable GPU')
    
    return parser.parse_args()

def process_image(image):
    target_size = 224
    width, height = image.size
    aspect_ratio = width / height
    if width > height:
        new_width = int(aspect_ratio * 256)
        new_height = 256
    else:
        new_width = 256
        new_height = int(256 / aspect_ratio)
    resized_image = image.resize((new_width, new_height))
    left = (new_width - target_size) / 2
    top = (new_height - target_size) / 2
    right = left + target_size
    bottom = top + target_size
    cropped_image = resized_image.crop((left, top, right, bottom))
    np_image = np.array(cropped_image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - mean) / std
    final_image = normalized_image.transpose((2, 0, 1))
    tensor_image = torch.from_numpy(final_image).float()
    return tensor_image

def translate(idx,idx_too_class,class_to_name):
    with open(class_to_name, 'r') as f:
        cat_to_name = json.load(f)
    topclasss = [idx_too_class.get(idx, 'Unknown') for idx in idx]
    top_classes = [cat_to_name[str(class_index)] for class_index in topclasss]
    return top_classes

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    plt.axis('off')
    plt.show()
    
    return ax

def display_result(ps,classes):
    y_pos = np.arange(len(classes))
    y_pos = y_pos[::-1]

    plt.barh(y_pos, ps, align='center')
    plt.yticks(y_pos, classes)
    plt.show()