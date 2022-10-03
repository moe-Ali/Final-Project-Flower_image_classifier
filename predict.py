import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import os
import numpy as np
import argparse
import json
from utility import loading_data,process_image,imshow

# Create Parse using ArgumentParser
parser = argparse.ArgumentParser(description="Trainning arguments")
# Create command line arguments using add_argument() from ArguementParser method
parser.add_argument('--img_path', type = str, default = './flowers/test/74/image_01200.jpg',        help="Path to the folder of flower images")
parser.add_argument('--checkpoint', type = str, default = './checkpoint.pth',                       help="Path to the checkpoint")
parser.add_argument('--top_k', type = int, default = 5,                                           help="top K most likely classes")
parser.add_argument('--category_names', type = str, default = './cat_to_name.json',                 help="Path to the folder of category_names")
parser.add_argument('--gpu',default=True,                                                         help="Will be using GPU for traning or not")
args = parser.parse_args()

if args.gpu==True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg11(pretrained=True)
        
    model.to(device)
    
    model.classifier = checkpoint['model_calssifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['model_class_to_idx']
    return model
def predict(image_path, model, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    img=process_image(image_path).view([1,3,224,224])
    img=img.to(device)
    with torch.no_grad():
        logps = model.forward(img)
    ps = torch.exp(logps)
    return ps.topk(topk)
def main():
    model=load_checkpoint(args.checkpoint)
    top_p,top_class=predict(args.img_path,model)
    my_dict = {y:x for x,y in model.class_to_idx.items()}
    array=[]
    for i in range(args.top_k):
        k=int(top_class[0][i].cpu().numpy())
        n=my_dict.get(k)
        n=int(n)
        array.append(n)
    lab=[cat_to_name[str(index)] for index in array]
    probability = top_p.cpu().numpy()
    print(f"the name of the highest probabilty : {lab[0]} with probabilty of {(probability[0][0]*100):.3f}% ")
    print(f"the names of top {args.top_k} : {lab}")
    #printing the approximation of probabilites of the top_k instade of using .astype(int)
    probability_app=[]
    for i in range(args.top_k):
        probability_app.append(f"{(probability[0][i]*100):.3f}%")
    print(f"the probabilities of top {args.top_k} : {probability_app}")
if __name__ == "__main__":
    main()


