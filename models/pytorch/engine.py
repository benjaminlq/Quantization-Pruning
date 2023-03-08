import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from config import LOGGER
import utils
from tqdm import tqdm
      
def train(model, dataloader, optimizer, criterion, args):
    trainloader = dataloader.train_dataloader()
    valloader = dataloader.val_dataloader()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor = 0.5, patience = 3
    )
    max_acc = 0
    device = torch.device(args.device)
    model.to(device)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        curr_loss = 0
        tk0 = tqdm(trainloader, total=len(trainloader))
        for i, data in enumerate(tk0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            curr_loss += loss.item()
            if i % 200 == 199:
                print(f"Epoch {epoch+1}, {i+1} loss: {curr_loss / 200: .3f}")
                curr_loss = 0
        
        val_loss, val_acc = evaluate(model, valloader, criterion, device)
        
        LOGGER.info(f"Epoch {epoch+1}: Train Loss = {epoch_loss/i: .3f}, Val Loss = {val_loss: .3f}, Test Accuracy = {val_acc:.3f}")
        scheduler.step(val_loss)
        
        if val_acc > max_acc:
            utils.save_model(model, args.ckpt_path)
            max_acc = val_acc
            LOGGER.info('Model Saved at: {} with test accuracy {}'.format(args.ckpt_path, max_acc))

def evaluate(model, testloader, criterion, device):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        total_no, correct_no = 0, 0
        tk0 = tqdm(testloader, total=len(testloader))
        for i, data in enumerate(tk0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total_no += labels.size(0)
            correct_no += (predicted == labels).sum().item()
            
    return epoch_loss/i, correct_no / total_no