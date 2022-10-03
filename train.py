import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import os
import numpy as np
import argparse
from utility import loading_data

# Create Parse using ArgumentParser
parser = argparse.ArgumentParser(description="Trainning arguments")
# Create command line arguments using add_argument() from ArguementParser method
parser.add_argument('--dir', type = str, default = './flowers',        help="Path to the folder of flower images")
parser.add_argument('--arch', type = str, default = 'vgg11',          help="Path to CNN Model Architecture")
parser.add_argument('--save_dir',type= str, default="checkpoint.pth", help="Set directory to save checkpoints")
parser.add_argument('--learning_rate', type=int,default=0.001,        help="Set learning rate")
parser.add_argument('--hidden_units1', type=int,default=500,          help="Set number of first hidden layer units")
parser.add_argument('--hidden_units2', type=int,default=250,          help="Set number of second hidden layer units")
parser.add_argument('--epochs',type=int,default=2,                    help="Set number 0f epochs")
parser.add_argument('--gpu',default=True,                             help="Will be using GPU for traning or not")
args = parser.parse_args()

data_transforms,image_datasets,dataloaders=loading_data(args.dir)
trainloader,validloader,testloader=dataloaders["train"],dataloaders["valid"],dataloaders["test"]
                    
def main():
    model = getattr(models, args.arch)(pretrained=True)
    if args.gpu==True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier=nn.Sequential(nn.Linear(25088,args.hidden_units1),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(args.hidden_units1,args.hidden_units2),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(args.hidden_units2,102),
                                   nn.LogSoftmax(dim=1))
    criterion= nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=args.learning_rate)
    model.to(device)
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
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
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"validation loss: {valid_loss/len(testloader):.3f}.. "
                      f"validation accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    checkpoint={'input_size': 25088,
                'output_size': 102,
                "model_calssifier" : model.classifier,
                "model_state_dict" : model.state_dict(),
                "epochs" : epoch,
                "optimizer_stare_dict":optimizer.state_dict,
                "model_class_to_idx":image_datasets['train'].class_to_idx
                }
    torch.save(checkpoint,args.save_dir)

# Call to main function to run the program
if __name__ == "__main__":
    main()
    