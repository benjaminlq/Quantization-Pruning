import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import config
import utils
      
def train(model, dataloader, optimizer, criterion, args):
    trainloader = dataloader.train_dataloader()
    valloader = dataloader.val_dataloader()
    max_acc = 0
    device = torch.device(args.device)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        curr_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            curr_loss += loss.item()
            if i % 200 == 199:
                print(f"Epoch {epoch+1}, {i+1} loss: {curr_loss / 200: .3f}")
                curr_loss = 0
        
        test_acc = evaluate(model, valloader, device)
        
        print(f"Epoch {epoch+1}: Loss = {epoch_loss/i: .3f}, Test Accuracy = {test_acc:.3f}")
    
        if test_acc > max_acc:
            utils.save_model(model, args.ckpt_path)
            print('Model Saved at: {} with test accuracy {}'.format(args.ckpt_path, max_acc))

def evaluate(model, testloader, device):
    model.eval()
    with torch.no_grad():
        total_no, correct_no = 0, 0
        for data in testloader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total_no += labels.size(0)
            correct_no += (predicted == labels).sum().item()
            
    return correct_no / total_no