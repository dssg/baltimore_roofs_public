from copy import deepcopy
from pathlib import Path
import random
import string
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import experiment_config as config

# import src.features.deepl as deepl
from src.features.data_set import InMemoryBlocklotDataset
from src.pipeline.image_standardizer import ImageStandardizer
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd


class TransferLearning:
    def __init__(
        self,
        batch_size,
        learning_rate=None,
        num_epochs=None,
        load_model=None,
        algorithm="Adam",
        pretrained="ResNet18",
        angle_variations=[0],
        optimizer=None,
        unfreeze=0,
        dropout=0.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.load_model = load_model
        self.pretrained = pretrained
        self.algorithm = algorithm
        self.angle_variations = angle_variations
        self.best_state = None
        self.optimizer = optimizer
        self.unfreeze = unfreeze
        self.dropout = dropout

        random_prefix = "".join(random.sample(string.ascii_letters, 4))
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.model_name = f"{now}_{random_prefix}"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.load_model:
            self.model = torch.load(self.load_model)
            self.model.to(self.device)
        else:
            model_class_name = self.pretrained.lower()
            model_class = getattr(models, model_class_name)
            weights_attr = getattr(models, f"{self.pretrained}_Weights")
            self.model = set_model(
                self.device, model_class(weights=weights_attr.DEFAULT), self.unfreeze, self.dropout
            )

    def checkpoint(self):
        return {
            "model": deepcopy(self.model.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
        }

    def to_save(self):
        return {
            "best_state": self.best_state,
            "current_state": self.checkpoint(),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "pretrained": self.pretrained,
            "algorithm": self.algorithm,
            "angle_variations": self.angle_variations,
            "optimizer": self.optimizer,
            "model_name": self.model_name,
            "unfreeze": self.unfreeze,
            "dropout": self.dropout,
        }

    @classmethod
    def load(cls, state):
        instance = cls(
            batch_size=state["batch_size"],
            learning_rate=state["learning_rate"],
            num_epochs=state["num_epochs"],
            pretrained=state["pretrained"],
            algorithm=state["algorithm"],
            angle_variations=state["angle_variations"],
            optimizer=state["optimizer"],
            unfreeze=state["unfreeze"],
            dropout=state["dropout"],
        )
        instance.model.load_state_dict(state["best_state"]["model"])
        instance.optimizer.load_state_dict(state["best_state"]["optimizer"])
        return instance

    def get_dataloaders(self, X, y):
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=config.seed
        )
        train = pd.DataFrame({"blocklot": train_X, "label": train_y})
        val = pd.DataFrame({"blocklot": test_X, "label": test_y})

        train_dataloader = get_dataloader(
            train, self.batch_size, angle_variations=self.angle_variations, shuffle=True
        )
        val_dataloader = get_dataloader(
            val, self.batch_size, angle_variations=self.angle_variations, shuffle=False
        )

        return {"train": train_dataloader, "val": val_dataloader}

    def log_to_tensorboard(self, phase, epoch, epoch_loss):
        logdir = str(
            Path(config.model_trainer.model_dir)
            / config.schema_prefix
            / "tensorboard"
            / self.model_name
        )
        hparam_dict = {
            "lr": self.learning_rate,
            "bsize": self.batch_size,
            "optimizer": self.optimizer.__class__.__name__,
            "pre-trained": self.pretrained,
            "unfreeze": self.unfreeze,
            "dropout": self.dropout,
        }

        metric_dict = {
            "hparam/loss": self.best_loss,
        }

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            w_hp.add_hparams(hparam_dict, metric_dict, run_name=f"/{logdir}")

    def fit(self, X, y):
        assert self.learning_rate is not None
        assert self.num_epochs is not None

        self.best_acc = 0.0
        self.best_loss = 99999999999
        self.criterion = nn.CrossEntropyLoss()
        if self.optimizer is None:
          optim_class = getattr(optim, self.algorithm)
          self.optimizer = optim_class(self.model.parameters(), self.learning_rate)

        dataloaders = self.get_dataloaders(X, y)

        for epoch in tqdm(range(self.num_epochs), desc="Epoch", leave=False):
            for phase in tqdm(["train", "val"], desc="Phase", leave=False):
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                batch_losses = []

                for inputs, labels in tqdm(
                    dataloaders[phase], desc="Batch", leave=False
                ):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                    batch_loss = loss.item()
                    batch_losses.append(batch_loss)

                epoch_loss = np.array(batch_losses).mean()

                if phase == "val" and epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.best_state = self.checkpoint()

                self.log_to_tensorboard(phase, epoch, epoch_loss)

        return self

    def forward(self, X: List[str]) -> dict[str, float]:
        df = pd.DataFrame({"blocklot": X})
        df["label"] = 0  # label is not used for prediction

        dataloader = get_dataloader(
            df, self.batch_size, angle_variations=[0], shuffle=False
        )

        self.model.eval()
        probs = []

        for inputs, _ in tqdm(dataloader, desc="Batch", leave=False):
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                probs_for_batch = torch.nn.functional.softmax(outputs, dim=1)
                probs.append(probs_for_batch)

        results = torch.cat(probs, dim=0).cpu().numpy()
        if len(X) != len(results):
            raise ValueError(
                f"Number of results ({len(results)}) "
                "does not match number of inputs ({len(X)})"
            )

        output = [float(damage_pred) for damage_pred in results[:, 1]]
        return dict(zip(X, output))

    def predict_proba(self, X: List[str]) -> dict[str, float]:
        return self.forward(X)


def get_dataloader(X, batch_size, angle_variations=[0], shuffle=False):
    dataset = InMemoryBlocklotDataset(
        X,
        transform=ImageStandardizer(output_dims=(224, 224)),
        angle_variation=angle_variations,
    )  # TODO: update transform?
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def set_model(device, pretrained, unfreeze=0, dropout=0.0):
    model = pretrained
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze == 1:
        for param in model.layer4[1].parameters():
            param.requires_grad = True

    if unfreeze == 2:
        for param in model.layer4.parameters():
            param.requires_grad = True

    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, 2))
    return model.to(device)
