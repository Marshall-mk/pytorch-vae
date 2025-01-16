import os
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
import torch
class MyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []  # List to hold all image file paths
        self.labels = []       # List to hold corresponding labels
        
        # Walk through directory, assuming class subdirectories
        for label, class_dir in enumerate(sorted(os.listdir(image_dir))):
            class_path = os.path.join(image_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(label)

        self.num_classes = len(set(self.labels))  # Determine the number of classes

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to fetch.
        Returns:
            dict: A dictionary with 'image' and 'label'.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open the image
        image = Image.open(image_path).convert("RGB")

        # Apply transform (if any)
        if self.transform:
            image = self.transform(image)

        # One-hot encode the label
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()

        return image, label
    
# class MyDataset(Dataset):
#     def __init__(self, image_dir, transform=None):
#         """
#         Args:
#             image_dir (str): Path to the directory containing images and labels.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.image_dir = image_dir
#         self.transform = transform
#         self.image_paths = []  # List to hold all image file paths
#         self.labels = []       # List to hold corresponding labels
        
#         # Walk through directory, assuming class subdirectories
#         for label, class_dir in enumerate(sorted(os.listdir(image_dir))):
#             class_path = os.path.join(image_dir, class_dir)
#             if os.path.isdir(class_path):
#                 for img_file in os.listdir(class_path):
#                     self.image_paths.append(os.path.join(class_path, img_file))
#                     self.labels.append(label)

#     def __len__(self):
#         """Returns the total number of samples in the dataset."""
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         """
#         Args:
#             idx (int): Index of the item to fetch.
#         Returns:
#             dict: A dictionary with 'image' and 'label'.
#         """
#         image_path = self.image_paths[idx]
#         label = self.labels[idx]

#         # Open the image
#         image = Image.open(image_path).convert("RGB")

#         # Apply transform (if any)
#         if self.transform:
#             image = self.transform(image)

#         return image, label


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        
#       =========================  CelebA Dataset  =========================
        mean = np.array([0.6104, 0.5033, 0.4965])
        std = np.array([0.2507, 0.2288, 0.2383])
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(224),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(224),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std),])

        self.train_dataset = MyDataset(self.data_dir+'3_class_train/', transform=train_transforms)
        self.val_dataset = MyDataset(self.data_dir+'3_class_val/', transform=val_transforms)
        print(f"Train size: {len(self.train_dataset)}")
        print(f"Val size: {len(self.val_dataset)}")
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     