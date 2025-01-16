import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from torchvision import datasets, transforms
import time
import os
import copy
import logging
from omegaconf.omegaconf import OmegaConf
import hydra
import timm
import torch.nn as nn
from torchvision import models
from torch import nn
import seaborn as sns
from typing import Optional, Sequence
from torch import Tensor
import torch.nn.functional as F
import warnings
from sklearn.metrics import classification_report
from torch.utils.data import random_split
warnings.filterwarnings("ignore")


class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

def focal_loss(
    alpha: Optional[Sequence] = None,
    gamma: float = 0.0,
    reduction: str = "mean",
    ignore_index: int = -100,
    device="cuda",
    dtype=torch.float32,
) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha, gamma=gamma, reduction=reduction, ignore_index=ignore_index
    )
    return fl

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

def calculate_accuracy(outputs, targets):
    preds = torch.sigmoid(outputs) > 0.5
    correct = (preds == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def mobilenet_v2(output=1):
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, output)
    return model

def convit_small(output=1):
    model = timm.create_model("convit_small", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model

def convnextv2_base(output=1):
    model = timm.create_model("convnextv2_base", pretrained=True)
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output)
    return model

def deit_base_patch16_224(output=1):
    model = timm.create_model("deit_base_patch16_224", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model

def efficientnetv2_l(output=1):
    model = timm.create_model("tf_efficientnetv2_l_in21k", pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, output)
    return model

def main():
    ########################################################## Dataset ##########################################################
    # prepare the data
    mean = np.array([0.6104, 0.5033, 0.4965])
    std = np.array([0.2507, 0.2288, 0.2383])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }
    use_train_split = False
    if use_train_split:
      train_dataset = datasets.ImageFolder(
        root=f"../../../dataset/train_reconstructed/", transform=data_transforms["train"]
      )
      val_fraction = 0.3
      train_size = int((1 - val_fraction) * len(train_dataset))
      val_size = len(train_dataset) - train_size
      train_datasetx, val_dataset = random_split(train_dataset, [train_size, val_size])
      # load data
      train_dataloader = torch.utils.data.DataLoader(
          train_datasetx, batch_size=32, shuffle=True, num_workers=10
      )
      val_dataloader = torch.utils.data.DataLoader(
          val_dataset, batch_size=32, shuffle=False, num_workers=10
      )
    else:
      train_dataset = datasets.ImageFolder(
          root=f"../../../dataset/train_reconstructed/", transform=data_transforms["train"]
      )
      val_dataset = datasets.ImageFolder(
          root=f"../../../dataset/val_reconstructed/", transform=data_transforms["val"]
      )
      # load data
      train_dataloader = torch.utils.data.DataLoader(
          train_dataset, batch_size=32, shuffle=True, num_workers=10
      )
      val_dataloader = torch.utils.data.DataLoader(
          val_dataset, batch_size=32, shuffle=False, num_workers=10
      )

    class_names = train_dataset.classes
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    model_name = 'all_models'
    class_counts = {"bcc": 1993, "mel": 2713, "scc":676}
    class_weights = []
    for i in range(len(class_names)):
        class_weights.append(1.0 / class_counts[class_names[i]])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights = class_weights / sum(class_weights)
    
    print(f"DataSet Sizes: {dataset_sizes}")
    print(f"Class Names: {class_names}")
    print(f"Class Weights: {class_weights}")
    print("Data Loaded")
    
    ########################################################## Model ##########################################################
    print("Multiple Models Training Started")
    all_models = [
        # "mobilenet",
        # "efficientnetv2_l",
        "convnextv2_base",
        # "convit_small",
        # "deit_base_patch16_224",
    ]
    for model_name in all_models:
        print(f"Training {model_name}")
        train_loss_type = "FocalLoss"
        optimizer_type = "Adam"
        scheduler_type = "ReduceLROnPlateau"
        # model selection
        if model_name == "mobilenet":
            model = mobilenet_v2(output=3)
        elif model_name == "efficientnetv2_l":
            model = efficientnetv2_l(output=3)
        elif model_name == "convnextv2_base":
            model = convnextv2_base(output=3)
        elif model_name == "convit_small":
            model = convit_small(output=3)
        elif model_name == "deit_base_patch16_224":
            model = deit_base_patch16_224(output=3)
        else:
            print("Please specify a valid model name")
        model.to(device)
        # criterion selection
        if train_loss_type == "FocalLoss":
            criterion = focal_loss(alpha=class_weights, gamma=2.0, reduction="mean")
        elif train_loss_type == "WeightedCELoss":
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # optimizer selection
        if optimizer_type == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif optimizer_type == "SGD":
            optimizer = optim.SGD(
                model.parameters(), lr=0.0001, momentum=0.9
            )
        elif optimizer_type == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
        elif optimizer_type == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        else:
            print("Please specify a valid optimizer")

        # scheduler selection
        if scheduler_type == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif scheduler_type == "MultiStepLR":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=[7, 10], gamma=0.1
            )
        elif scheduler_type == "ExponentialLR":
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=0, last_epoch=-1
            )
        else:
            print("Please specify a valid scheduler")

        since = time.time()

        print("Training Started")
        sys.stdout.flush()
        # Early stopping
        early_stopping = EarlyStopper(patience=10, min_delta=10)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = np.inf
        num_epochs = 50

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("=" * 30)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                    dataloader = train_dataloader
                else:
                    model.eval()
                    dataloader = val_dataloader

                running_loss = 0.0
                running_accuracy = 0.0

                all_labels = []
                all_preds = []

                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_accuracy += torch.sum(preds == labels.data)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_accuracy = running_accuracy.double() / dataset_sizes[phase]
                epoch_kappa = cohen_kappa_score(all_labels, all_preds)
                # Save training metrics
                if not os.path.exists(f"../metrics64by64/{model_name}_{train_loss_type}_{optimizer_type}_{scheduler_type}_train_metrics.csv"):
                    pd.DataFrame(
                        columns=[
                            "Epoch",
                            "Accuracy",
                            "Loss",
                            "Kappa",
                        ]
                    ).to_csv(
                        f"../metrics64by64/{model_name}_{train_loss_type}_{optimizer_type}_{scheduler_type}_train_metrics.csv",
                        index=False,
                    )
                pd.DataFrame(
                    [
                        [
                            epoch + 1,
                            epoch_accuracy.item(),
                            epoch_loss,
                            epoch_kappa,
                        ]
                    ],
                    columns=[
                        "Epoch",
                        "Accuracy",
                        "Loss",
                        "Kappa",
                    ],
                ).to_csv(
                    f"../metrics64by64/{model_name}_{train_loss_type}_{optimizer_type}_{scheduler_type}_train_metrics.csv",
                    mode="a",
                    header=False,
                    index=False,
                )
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f} Kappa: {epoch_kappa:.4f}")
                sys.stdout.flush()

                # Deep copy the model
                if phase == "val" and (
                    epoch_accuracy > best_acc or epoch_loss < best_loss
                ):
                    best_acc = epoch_accuracy
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save the best model
                    if not os.path.exists("../checkpoints64by64"):
                        os.makedirs("../checkpoints64by64")
                    torch.save(
                        model.state_dict(),
                        f"../checkpoints64by64/{model_name}_{train_loss_type}_{optimizer_type}_{scheduler_type}_best_model.pth",
                    )
                    logging.info(f"Best model checkpoint saved at epoch {epoch + 1}!")

                # Compute and plot confusion matrix for validation phase
                if phase == "val":
                    cm = confusion_matrix(all_labels, all_preds)
                    print(
                        classification_report(
                            all_labels, all_preds, target_names=class_names
                        )
                    )
                    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
                    precision = np.mean([cm[i, i] / np.sum(cm[:, i]) for i in range(3)])
                    recall = np.mean([cm[i, i] / np.sum(cm[i, :]) for i in range(3)])
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    val_kappa = cohen_kappa_score(all_labels, all_preds)
                    # log metrics
                    logging.info(
                        f"Epoch {epoch + 1}  Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1-Score: {f1_score:.4f} Loss: {epoch_loss:.4f} Val Kappa: {val_kappa:.4f}"
                    )
                    # save metrics
                    if not os.path.exists("../metrics64by64"):
                        os.makedirs("../metrics64by64")
                    if not os.path.exists(
                        f"../metrics64by64/{model_name}_{train_loss_type}_{optimizer_type}_{scheduler_type}_metrics.csv"
                    ):
                        pd.DataFrame(
                            columns=[
                                "Epoch",
                                "Accuracy",
                                "Precision",
                                "Recall",
                                "F1-Score",
                                "Loss",
                                "Kappa",
                            ]
                        ).to_csv(
                            f"../metrics64by64/{model_name}_{train_loss_type}_{optimizer_type}_{scheduler_type}_metrics.csv",
                            index=False,
                        )
                    pd.DataFrame(
                        [
                            [
                                epoch + 1,
                                accuracy,
                                precision,
                                recall,
                                f1_score,
                                epoch_loss,
                                val_kappa,
                            ]
                        ],
                        columns=[
                            "Epoch",
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "F1-Score",
                            "Loss",
                            "Kappa",
                        ],
                    ).to_csv(
                        f"../metrics64by64/{model_name}_{train_loss_type}_{optimizer_type}_{scheduler_type}_metrics.csv",
                        mode="a",
                        header=False,
                        index=False,
                    )
                    try:
                        plot_confusion_matrix(
                            cm,
                            classes=class_names,
                            title=f"{phase} Confusion Matrix",
                        )
                        plt.savefig(
                            f"../metrics64by64/{model_name}_{train_loss_type}_{optimizer_type}_{scheduler_type}_confusion_matrix_{epoch + 1}.png"
                        )
                    except Exception as e:
                        logging.error(f"Error while plotting confusion matrix: {e}")
            # Update the scheduler
            if scheduler is not None:
                if phase == "val":
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

            # Check if early stopping criteria is met
            if early_stopping.early_stop(epoch_loss):
                print("Early stopping triggered. Training stopped.")
                sys.stdout.flush()
                break
        time_elapsed = time.time() - since
        torch.save(
            model.state_dict(),
            f"../checkpoints64by64/{model_name}_{train_loss_type}_{optimizer_type}_{scheduler_type}_last_model.pth",
        )
        logging.info("Last model checkpoint saved!")
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))
        print("Best val Loss: {:4f}".format(best_loss))
        sys.stdout.flush()
        # load best model weights
        model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    main()
