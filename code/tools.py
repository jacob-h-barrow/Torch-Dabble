import torch
import os

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataclasses import dataclass
import sys

sys.path.append('/home/parzival/Desktop/DL_2023/')

from proc_gpu import GpuResourceUse, GpuHandler

@dataclass
class TorchData:
    train: torch.utils.data.dataloader.DataLoader
    test: torch.utils.data.dataloader.DataLoader
    
def create(func):
    def create_folder(*args, **kwargs):
        where = args[1]
        
        if os.path.exists(where):
            if not os.path.isdir(where):
                raise Exception(f'{where} exists but isnt a dir')
        else:
            os.mkdir(where)
        
        return func(*args, **kwargs)
    return create_folder

@create
def download_dataset(name: str, where: str, train: bool=True, download: bool=True):
    match name:
        case 'FashionMNIST':
            return datasets.FashionMNIST(
                root=where,
                train=train,
                download=download,
                transform=ToTensor(),
            )
        case _:
            raise Exception(f'Cant do {name} dataset right now')
            
def create_loader(data, batch_size: int=64):
    return DataLoader(data, batch_size=batch_size)
    
def get_device(_print: bool=True):
    match torch.cuda.is_available():
        case True:
            res = 'cuda'
        case _:
            res = 'cpu'
            
    if _print:
        print(f'Using a {res} device')
        
    return res
    
class TorchHandler:
    def __init__(self, model, loss_fxn: str = 'CEL', optimizer: str = 'SGD', lr: float = 1e-3):
        # Fix later
        self.gpus = GpuHandler()
        
        device = get_device(_print=False)
        self.model = model
        
        match loss_fxn:
            case 'CEL':
                self.loss_fxn = nn.CrossEntropyLoss()
            case _:
                self.raise_ex(f'{loss_fxn} not supported')
        
        match optimizer:
            case 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
            case _:
                self.raise_ex(f'{optimizer} not supported')
        
    def raise_ex(self, message: str):
        raise Exception(message)
        
    def train(self, dataloader, epoch_print: int = 100, epoch_stop: int = 100000):
        done = False
        size = len(dataloader.dataset)
        res = {}
        
        res['training'] = [self.gpus.pull()]
        self.model.train()
        
        while epoch_stop and not done:
            res['training'].append(self.gpus.pull()]
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                
                pred = self.model(X)
                loss = self.loss_fxn(pred, y)
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if batch % epoch_print == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')
            
            done = True
            
        return res
                    
    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        
        self.model.eval()
        
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                
                pred = self.model(X)
                
                test_loss += self.loss_fxn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= num_batches
        correct /= size
        
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
def train_validate(data: TorchData, handler: TorchHandler, epoch_runs: int = 5, epoch_print: int = 300, print_train: bool = True):
    for i in range(epoch_runs):
        print(f"Epoch {i+1}\n-------------------------------")
        if (res := handler.train(data.train, epoch_print=epoch_print)) and print_train:
            print(f'Training consumed these resources!:\n{res}')
        
        handler.test(data.test)

if __name__ == "__main__":
    from neural_designs import NeuralNetwork
    from utils import export_state_dict_to_numpy, export_state_dict_to_json, load_state_dict_from_json, test_state_dict_eq
    
    import os
    
    device = get_device()
    
    model = NeuralNetwork().to(device)
    
    if not os.path.exists('./model.pth'):
        download = False if os.path.exists('./data/FashionMNIST/') else True
        
        training_data = download_dataset('FashionMNIST', './data', download=download)
        test_data = download_dataset('FashionMNIST', './data', train=False, download=download)
        
        train_loader, test_loader = [create_loader(data) for data in (training_data, test_data)]
        
        for X, y in test_loader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
        
        fashion_mnist_data = TorchData(train_loader, test_loader)
        
        handler = TorchHandler(model)
    
        train_validate(fashion_mnist_data, handler)
    else:
        print("Skipping Training!")
        model.load_state_dict(torch.load('model.pth'))
        
        handler = TorchHandler(model)
    
    export = export_state_dict_to_numpy(handler.model)
    sd_json = export_state_dict_to_json(export)
    sd_loaded = load_state_dict_from_json(sd_json)
    
    print(f'Testing equality, passed: {test_state_dict_eq(sd_json, handler.model)}')
    
    # torch.save(model.state_dict(), 'model.pth')
