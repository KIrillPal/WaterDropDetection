# This file contains operations with checkpoint files
#
# Checkpoint contains model parameters, optimizer and loss function,
# Losses of the train and validation dataset
#

def last_checkpoint(checkpoint_dir):
    from pathlib import Path
    from os.path import getctime
    files = list(Path(checkpoint_dir).glob('*.pt'))
    assert files, f"No available checkpoints in '{checkpoint_dir}'"
    return max(files, key=getctime)


def print_checkpoint(checkpoint):
    print("Epochs: ", checkpoint["epochs"])
    print("Train loss: ", checkpoint["train_loss"])
    print("Valid loss: ", checkpoint["val_loss"])


def get_checkpoint(
    model, 
    mode, 
    optim, 
    loss_fn, 
    epoch, 
    train_loss, 
    val_loss, 
    iou,        
    acc,
    prec,
    recall
):
    # Creating the name of checkpoint
    name = model.__class__.__name__ + mode \
        + loss_fn.__class__.__name__[:3]   \
        + 't' + str(int(train_loss * 100)) \
        + 'v' + str(int(val_loss * 100))   \
        + 'e' + str(epoch)                 \
        + '.pt'
    return {
        "name": name,
        "epochs": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "iou": iou,
        "accuracy": acc,
        "precision": prec,
        "recall": recall
    }


def save_checkpoint(checkpoint, dir, name):
    from pathlib import Path
    from torch import save
    path = Path(dir, name)
    save(checkpoint, path)
    print(f"Progress saved to '{path}'")


def clear_checkpoints(checkpoint_dir, save_last=-1, excepts=[], condition=None):
    from pathlib import Path
    from os.path import getctime
    to_remove = list(Path(checkpoint_dir).glob('*.pt'))

    # Select files by condition
    if condition is not None:
        from torch import load
        to_remove = [f for f in to_remove if condition(load(f))]
    # Save last elems
    if (save_last > 0):
        to_remove = sorted(to_remove, key=getctime)[:-save_last]
    # Save exceptions
    to_remove = [f for f in to_remove if f not in excepts]

    if not to_remove:
        print("Nothing to remove")
    
    for path in to_remove:
        path.unlink()
        print("Removed", path)

def print_checkpoint(checkpoint):
    print("Train BCE loss:", checkpoint["train_loss"])
    print("Valid BCE loss:", checkpoint["val_loss"])
    print("IoU:      ", checkpoint["iou"], '\n')
    print("Accuracy: ", checkpoint["accuracy"])
    print("Precision:", checkpoint["precision"])
    print("Recall:   ", checkpoint["recall"], '\n')
