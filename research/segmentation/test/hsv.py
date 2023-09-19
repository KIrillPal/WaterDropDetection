import sys
sys.path.append('/beta/students/kondrashov/WaterDropDetection/research/segmentation')

import torch
from checkpoints import *
from measures import *

config = {
    "lr": 0.004,
    "batch_size": 16,
    "epochs": 150,
    "threshold": 0.3,
    "init_from_checkpoint": False,
    "input_mode": "HSV",
    "milestones": [10, 30, 50, 70, 85],
    "gamma":  0.63,
    "image_dir": '../../data/stereo/train/image',
    "mask_dir":  '../../data/stereo/train/mask',
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints",
    "checkpoint": None, # if None loads last saved checkpoint
    "print_model": False,
    "binarization": True,
    "seed": None # if None uses random seed,
}

config["channels"] = len(config["input_mode"]) - int('I' in config["input_mode"])
print(f"Training using {config['device']}")

# Set seed
if config["seed"] is not None:
    seed = config["seed"]
else:
    seed = torch.random.seed()
torch.manual_seed(seed)    
print('Seed', seed)

def random_split(dataset, val_percent=0.15, test_percent=0.15):
    val_size = int(len(dataset) * val_percent)
    test_size = int(len(dataset) * test_percent)
    train_size = len(dataset) - val_size - test_size
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Load train and val datasets and prepare loaders

from torch.utils.data import DataLoader
import importlib
import dataset
importlib.reload(dataset)
from dataset import WaterDropDataset
dataset = WaterDropDataset(
    mode=config["input_mode"],
    image_dir=config["image_dir"],
    mask_dir=config["mask_dir"],
    binarization = config["binarization"],
    threshold=config["threshold"],
    crop_shape=(256, 256)
)

assert dataset, "Dataset is empty!"

train_dataset, val_dataset, test_dataset = random_split(dataset, 0.1, 0.1)
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

print (f'Loaded {len(dataset)} images\n')
print (f'Train: {len(train_dataset)} images, {len(train_loader)} batches')
print (f'Val:   {len(val_dataset)} images, {len(val_loader)} batches')
print (f'Test:  {len(test_dataset)} images, {len(test_loader)} batches')


# Load model, loss function and optimizer
from torch import nn
import importlib
import unet

importlib.reload(unet)
from unet import UNet, RUNet
from unet import init_weights
from pathlib import Path
from torch.optim import lr_scheduler

extras = ('I' if 'I' in config["input_mode"] else '')
model = UNet(config["channels"], extras=extras).to(config['device'])
optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"])

# Load or fill weights
# And set the start_epoch of model
best_checkpoint = None
if config["init_from_checkpoint"]:
    # Load checkpoint
    if config["checkpoint"] is None:
        path = last_checkpoint(config["checkpoint_dir"])
    else:
        path = Path(config["checkpoint_dir"], config["checkpoint"])
    checkpoint = torch.load(path)
    best_checkpoint = checkpoint

    # Load model & optim from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded parameters from '{path}'")
    print_checkpoint(checkpoint)
    start_epoch = checkpoint["epochs"]
else:
    init_weights(model, torch.nn.init.uniform_, a = -1., b = 1.)
    print("Randomly initiated parameters")
    start_epoch = 0

scheduler = lr_scheduler.MultiStepLR(
    optimizer,
    milestones=config["milestones"],
    gamma=config["gamma"]
)

loss_fn = torch.nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

layers = model.train()
if config["print_model"]:
    print(layers)

def to_device(x, y):
    x = x.to(config['device'])
    y = y.to(config['device'])
    return x, y


def validate(model, loss_fn, loader=val_loader):
    import math
    model.eval()

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    ious = []
    with torch.no_grad():
        for x, y in loader:
            x, y = to_device(x, y)
            pred = model(x)
            loss = loss_fn(pred, y)

            if not math.isnan(loss.item()):
                losses.append(loss.item())
            else:
                print("Nan on validation")

            pred = torch.sigmoid(model(x))
            pred = pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            if config["binarization"]:
                pred = (pred >= config["threshold"])

            accuracies.append(accuracy(y, pred))
            ious.append(IoU(y, pred))
            precisions.append(precision(y, pred))
            recalls.append(recall(y, pred))
    model.train()
    mean = lambda l: sum(l) / len(l) if len(l) > 0 else -1
    return mean(losses), mean(ious), mean(accuracies), mean(precisions), mean(recalls)

def ch_score(checkpoint):
    return checkpoint["iou"]

if not config["init_from_checkpoint"]:
    best_checkpoint = {
        "name": "nullcheck",
        "epochs": 0,
        "model_state_dict": [],
        "optimizer_state_dict": [],
        "train_loss": 1e100,
        "val_loss": 1e100,
        "iou": 0,
        "accuracy": 0,
        "precision": 0,
        "recall": 0
    }

from tqdm import tqdm
from sys import stdout
import math
import wandb
from datetime import datetime

# Start wandb
wandb.init(
    name= "REAL_" + config["input_mode"] + "_" + str(config["lr"]) + "_" + str(datetime.now()),
    project="water-drop-detection",
    config={
    "learning_rate": config["lr"],
    "architecture": "UNet",
    "dataset": "Stereo",
    "seed": seed,
    "epochs": config["epochs"],
    "checkpoint": best_checkpoint
    }
)
print('Seed', seed)

def train(save_checkpoints=True, lr=None):
    # If lr=None, learning rate is used from optimizer
    if lr is not None:
        optimizer.param_groups[0]["lr"] = lr
    global best_checkpoint

    for epoch in range(config['epochs']):
        epoch += start_epoch
        print("Epoch", epoch, "| lr", optimizer.state_dict()["param_groups"][0]["lr"])

        loader = tqdm(train_loader)
        losses = []

        # Training this epoch
        for image, gt in loader:
            image, gt = to_device(image, gt)
            with torch.cuda.amp.autocast():
                pred = model(image)
                if math.isnan(torch.max(pred)):
                    print(f"NaN pred occured")
                loss = loss_fn(pred, gt)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss = loss.item()
            if math.isnan(train_loss):
                print(f"NaN loss occured")
                return

            losses.append(train_loss)
            loader.set_postfix(loss=train_loss)

        optimizer.step()
        scheduler.step()

        # Compute metrics
        train_loss = sum(losses) / len(losses)
        checkpoint = get_checkpoint(
            model,
            config["input_mode"],
            optimizer,
            loss_fn,
            epoch,
            train_loss,
            *validate(model, loss_fn)
        )
        print_checkpoint(checkpoint)

        wandb.log({"checkpoint": checkpoint})

        if save_checkpoints:
            save_checkpoint(
                checkpoint,
                config["checkpoint_dir"],
                checkpoint["name"]
            )
        # Find best checkpoint
        elif ch_score(best_checkpoint) < ch_score(checkpoint):
            best_checkpoint = checkpoint

train(save_checkpoints=False)
start_epoch = start_epoch + config["epochs"]
print_checkpoint(best_checkpoint)
save_checkpoint(
                best_checkpoint,
                config["checkpoint_dir"],
                best_checkpoint["name"]
            )

def test(checkpoint):
        print("Name:", checkpoint["name"])
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, iou, acc, prec, rec = validate(model, loss_fn, test_loader)
        checkpoint["iou"] = iou
        checkpoint["accuracy"] = acc
        checkpoint["precision"] = prec
        checkpoint["recall"] = rec
        print("Test loss: ", test_loss)
        print_checkpoint(checkpoint)
        print("\n")

test(best_checkpoint)

wandb.finish()