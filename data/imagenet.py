from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config.config as config
import os.path as osp

class ImageNetEvalLoader:
    def __init__(
        self,
    ):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        self.dataset = datasets.ImageNet(root = osp.join(config.DATA_PATH, "raw"),
                                         split = "val", transforms=self.transform,
                                         download= True)
        
    def get_dataloader(
        self,
        batch_size,
    ):
        return DataLoader(
            self.dataset,
            batch_size = batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = config.NUM_WORKERS,
            pin_memory=True
        )

if __name__ == "__main__":
    dataloader = ImageNetEvalLoader()
    val_loader = dataloader.get_dataloader(batch_size=16)
    sample_batch = next(iter(val_loader))
    print(sample_batch.size())
    