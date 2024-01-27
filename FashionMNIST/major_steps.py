import torch
from torch import nn
def eval_model(model: torch.nn.Module,
              data_loader : torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = None):
    loss, acc = 0,0
    model.to(device)
    model.eval()# Put model in eval mode
    # turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1.forward pass
            y_pred = model(X)
            
            # 2.Calculate loss and accuracy
            loss+=loss_fn(y_pred,y)
            acc+=accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        #adjust the metrics
        loss /= len(data_loader)
        acc /= len(data_loader)
    return{"model_name": model.__class__.__name__,
          "model_loss": loss.item(),
          "model_acc": acc}
def train_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              accuracy_fn,
              device: torch.device=None):
    train_loss,train_acc=0,0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y=X.to(device), y.to(device)
        
        # 1. forward pass
        y_pred = model(X)
        
        # 2 . Calucate loss
        loss=loss_fn(y_pred,y)
        train_loss+=loss
        train_acc+=accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))# go from logits to pred labels
        
        # 3 . optimizer zero grad
        optimizer.zero_grad()
        
        # 4 . Loss backwards
        loss.backward()
        
        # 5 . Optimizer step
        optimizer.step()
        
    #Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} || Train accuracy : {train_acc:.2f}%")
    
def test_step(data_loader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             loss_fn: torch.nn.Module,
             accuracy_fn,
             device: torch.device = None):
    test_loss, test_acc = 0,0
    model.to(device)
    model.eval()# Put model in eval mode
    # turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1.forward pass
            test_pred = model(X)
            
            # 2.Calculate loss and accuracy
            test_loss+=loss_fn(test_pred,y)
            test_acc+=accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        #adjust the metrics
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


def results_model1():
    return {'model_name': 'FashionMNISTModelV1',
        'model_loss': 0.6850008368492126,
        'model_acc': 75.01996805111821}