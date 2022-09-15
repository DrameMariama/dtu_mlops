import argparse
import sys
from time import sleep

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print(args.command)
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.003)
        parser.add_argument("--epochs", default=10, help="number of epochs to train the model")
        parser.add_argument("--model_path", default="./model.pt", help="the path to save model checkpoint")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        criterion  = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), args.lr)
        for epoch in range(int(args.epochs)):
            running_loss = 0.0
            counter = 0
            with tqdm.tqdm(train_loader, unit="batch") as tepoch:
                for images, labels in tepoch:
                    tepoch.set_description(f"Epoch{epoch}")
                    images = images.view(images.shape[0], -1).float()
                    optimizer.zero_grad()
                    output = model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    counter += 1
                    tepoch.set_postfix(loss=(running_loss / (counter * train_loader.batch_size)))
                    #tepoch.set_postfix(loss=loss.item())
                    sleep(0.1)
        torch.save(model.state_dict(), args.model_path)
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Evaluation arguments')
        parser.add_argument('--load_model_from', default="./model.pt")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        state_dict = torch.load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(state_dict)
        _, test_set = mnist()
        test_loader = DataLoader(test_set, batch_size=64)
        true_predictions = 0
        with tqdm.tqdm(test_loader, unit="batch") as tepoch:
            
            for images, labels in tepoch:
                images = images.view(images.shape[0], -1)
                output = model(images.float())
                probs = nn.functional.softmax(output, dim=1)
                _, top_class = probs.topk(1, dim=1)
#                 print(top_class.shape)
                equals = top_class == labels.view(*top_class.shape)
#                 print(equals.type(torch.FloatTensor).sum())
                true_predictions += equals.type(torch.FloatTensor).sum().item()
#               print(true_predictions)
                sleep(0.1)
        accuracy = true_predictions/(len(test_set))  
        print(f"Evaluation accuracy is {accuracy*100:.2f} %")

if __name__ == '__main__':
    TrainOREvaluate()


    
    
    
    
    
    
    
    