from __future__ import print_function, division
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


# import numpy as np
# import torchvision
from torchvision import models

# import matplotlib.pyplot as plt
import pandas as pd

# from PIL import Image
import time
import copy
import pickle

import tqdm as tq
from src.features.data_set import BlocklotDataset
from src.features.data_splitter import get_split_data
from src.features.image_standardizer import ImageStandardizer
from src.db import run_query
from src.features.labeler import labeler

# from src.features.image_variations import tensor_to_numpy
from src.config import experiment_config


def get_missing_blocklots():

    with open("data/interim/missing.pkl", "rb") as f:
        return pickle.load(f)


def get_labeled_data():
    # New Query
    data = run_query(
        """
            SELECT
                lb.blocklot,
                mainroofpc AS roof_damage_score,
                labeled_inters
            FROM
                processed.labeled_blocklots lb
            LEFT JOIN raw.roofdata_2018 r ON
                lb.blocklot = r.blocklot
        """
    )
    df = pd.DataFrame(data, columns=data[0].keys())

    # Cleaning data function
    # Filling NAs in roof_damage_score with 0 so labeler will exclude it
    df["roof_damage_score"].fillna(0, inplace=True)

    # Creating dataframe with only blocklots (that contain images) and labels
    df_labeled = labeler(
        df, damage_columns=["roof_damage_score"]
    )  # TODO: rewrite labeler

    missing_blocklots = get_missing_blocklots()
    df_labeled = df_labeled[~df_labeled["blocklot"].isin(missing_blocklots)]

    return df_labeled[["blocklot", "label"]]


def get_resampled_data(df, n=1000, random_state=experiment_config.random_seed):
    return (
        df.groupby("label")
        .apply(lambda x: x.sample(n=1000, random_state=random_state))
        .reset_index(drop=True)
    )


def get_blocklot_dataloader(
    df,
    transform=ImageStandardizer((224, 224)),
    batch_size=experiment_config.train.batch_size,
):
    dataset = BlocklotDataset(df, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# The actual function of the model
def train_model(
    dataloaders,
    model,
    criterion,
    optimizer,
    device,
    writer,
    num_epochs=10,
    scheduler=None,
    model_name="unnamed_model",
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 99999999999

    for epoch in tq.tqdm(range(num_epochs), desc="Epoch", position=0):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in tq.tqdm(["train", "val"], desc="Phase", position=1):
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tq.tqdm(dataloaders[phase], desc="Batch", position=2):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train" and scheduler:
                writer.add_scalar("learning_rate", scheduler.get_last_lr())
                scheduler.step()

            dataset_size = len(dataloaders[phase]) * dataloaders[phase].batch_size
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"Acc/{phase}", epoch_acc, epoch)

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"model_{model_name}.pt")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def set_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)

    return model.to(device)


if __name__ == "__main__":

    model_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Tensorboard writer
    writer = SummaryWriter(f"{experiment_config.models_path}/{model_name}")
    print(f"{experiment_config.models_path}/{model_name}")

    # Splitting the set
    resampled_data = get_resampled_data(get_labeled_data())
    train, val = get_split_data(resampled_data)
    logging.info(f"Number of damaged labels in train data: {(train.label==1).sum()}")
    logging.info(f"Number of damaged labels in validation data: {(val.label==1).sum()}")

    # Data loaders
    train_loader = get_blocklot_dataloader(train)
    valid_loader = get_blocklot_dataloader(val)
    dataloaders = {"train": train_loader, "val": valid_loader}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = set_model(device)

    # Train parameters
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(
        model.parameters(),
        experiment_config.train.learning_rate,
    )

    # Train and evaluate
    train_model(
        dataloaders,
        model,
        criterion,
        optimizer,
        device,
        writer,
        num_epochs=experiment_config.train.num_epochs,
        model_name=model_name,
        # scheduler=exp_lr_scheduler,
    )

    torch.save(model, f"model_{model_name}_finished.pt")
