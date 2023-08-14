import torch
from checkpoints import *
from measures import *
config = {
    "lr": 0.002,
    "batch_size": 8,
    "epochs": 80,
    "threshold": 0.42,
    "init_from_checkpoint": False,
    "input_mode": "GS",
    "image_dir": '../../data/stereo/train/image',
    "mask_dir": '../../data/stereo/train/mask',
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints",
    "checkpoint": "UNetAdamBCEt39v40e50.pt", # if None loads last saved checkpoint
    "print_model": False,
    "binarization": True,
    "seed": 1301 # if None uses random seed
}
config["channels"] = len(config["input_mode"])
print(f"Training {config['input_mode']} using {config['device']}")

# Set seed
if config["seed"] is not None:
    torch.manual_seed(config["seed"])  


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

train_dataset, val_dataset = dataset.random_split(0.1)
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

print (f'Loaded {len(dataset)} images\n')
print (f'Train: {len(train_dataset)} images, {len(train_loader)} batches')
print (f'Val: {len(val_dataset)} images, {len(val_loader)} batches')


# Load model, loss function and optimizer

from torch import nn
from unet import UNet
from unet import init_weights
from pathlib import Path
from torch.optim import lr_scheduler

model = UNet(config["channels"]).to(config['device'])
optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"])

# Load or fill weights
# And set the start_epoch of model
if config["init_from_checkpoint"]:
    # Load checkpoint
    if config["checkpoint"] is None:
        path = last_checkpoint(config["checkpoint_dir"])
    else:
        path = Path(config["checkpoint_dir"], config["checkpoint"])     
    checkpoint = torch.load(path)
    
    # Load model & optim from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded parameters from '{path}'")
    print_checkpoint(checkpoint)
    start_epoch = checkpoint["epochs"]
else:
    init_weights(model, torch.nn.init.normal_, mean=0., std=1)
    print("Randomly initiated parameters")
    start_epoch = 0

scheduler = lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[
        3, 7, 11, 15, 
        20, 25, 30, 36, 
        42, 48, 55, 62, 70
    ],
    gamma=0.8
)

loss_fn = torch.nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

layers = model.train()
if config["print_model"]:
    print(layers)


def validate(model, loss_fn):
    model.eval()

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    ious = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(config['device'])
            y = y.to(config['device'])

            pred = model(x)
            loss = loss_fn(pred, y)
	    l = loss.item()
	    if l == l:
            	losses.append(l)
            
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
    mean = lambda l: sum(l) / len(l)
    return mean(losses), mean(ious), mean(accuracies), mean(precisions), mean(recalls)


def ch_score(checkpoint):
    return (2 - checkpoint["train_loss"] - checkpoint["val_loss"]) / 2


from tqdm import tqdm
from sys import stdout
def train(save_checkpoints=True):
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
    
    for epoch in range(config['epochs']):
        epoch += start_epoch
        print("Epoch", epoch, "| lr", optimizer.state_dict()["param_groups"][0]["lr"])
        
        loader = tqdm(train_loader)
        losses = []

        # Training this epoch
        for image, gt in loader:
            image = image.to(config['device'])
            gt = gt.float().to(config['device'])
            with torch.cuda.amp.autocast():
                pred = model(image)
                loss = loss_fn(pred, gt)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            train_loss = loss.item()
            if train_loss != train_loss:
                print(f"NaN loss occured")
                return best_checkpoint
            
            losses.append(train_loss)
            loader.set_postfix(loss=train_loss)
            scaler.step(optimizer)
            scaler.update()
            
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
        
        if save_checkpoints:
            save_checkpoint(
                checkpoint,
                config["checkpoint_dir"],
                checkpoint["name"]
            )
        # Find best checkpoint
        elif ch_score(best_checkpoint) < ch_score(checkpoint):
            best_checkpoint = checkpoint
    return best_checkpoint


best_checkpoint = train(save_checkpoints=False)
print_checkpoint(best_checkpoint)
