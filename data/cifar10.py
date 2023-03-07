from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config
import os.path as osp
from typing import Tuple

class CIFAR10DataLoader:
    def __init__(
        self,
        batch_size: int = 32,
        input_size: Tuple[int, int, int] = (3, 224, 224)
    ):
        self.batch_size = batch_size
        self.C, self.H, self.W = input_size
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(
                    size=(self.H,self.W), scale=(0.90, 1.00), ratio=(0.90, 1.10)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                
            ]
        )        

        self.val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.Resize(224),
            ]
        )  
        
        self.train_dataset = datasets.CIFAR10(root = osp.join(config.DATA_PATH, "raw"),
                                              train = True, transform=self.train_transforms, download = True)
        self.val_dataset = datasets.CIFAR10(root = osp.join(config.DATA_PATH, "raw"),
                                            train = False, transform=self.val_transforms, download = True)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = config.NUM_WORKERS,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = config.NUM_WORKERS,
            pin_memory=True
        )
        
if __name__ == "__main__":
    dataloader = CIFAR10DataLoader(batch_size=2)
    valloader = dataloader.val_dataloader()
    sample_imgs, sample_labels = next(iter(valloader))
    sample_imgs = sample_imgs.detach().cpu().numpy()
    print(sample_imgs)