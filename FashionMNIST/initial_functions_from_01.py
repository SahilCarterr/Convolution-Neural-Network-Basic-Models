def data_call():
    """
    Returns
    train_data,test_data,train_dataloader,test_dataloader

    """
    #Import torchvision
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    #Import matplotlib for visualization

    #Setup training Data
    train_data=datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=None
    )
    test_data=datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    from torch.utils.data import DataLoader

    #Setup the batch size hyperparameter
    BATCH_SIZE=32

    #Turn Data Set into iterables(Batches)
    train_dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)

    return train_data,test_data,train_dataloader,test_dataloader

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
from timeit import default_timer as timer
import torch
def print_train_time(start: float, end: float , device: torch.device=None):
    """Prints difference between start and end time.
    Args:
        start (float):Start time of computation (preferred in timeit format)
        end (float) : End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None
        
    Returns:
        float : time between start and end in seconds (higher is longer).
    """
    total_time=end-start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def results_model_0():
    return{'model_name': 'FashionMNISTModelV0',
    'model_loss': 0.4766390025615692,
    'model_acc': 83.42651757188499}