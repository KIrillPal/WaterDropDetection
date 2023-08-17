# File with main measures for binary segmentation
# Working with ndarrays

def nan_to_num(x, num=0):
    return x if x == x else num
         

def accuracy(y_true, y_pred):
    mean = 0
    for y, z in zip(y_true, y_pred):
        intersection = 1 - (y - z) ** 2
        acc = intersection.sum() / y.size
        mean += acc
    return mean / len(y_true)


def precision(y_true, y_pred):
    mean = 0
    for y, z in zip(y_true, y_pred):
        if y.sum() == 0:
            continue
        acc = (y * z).sum() / y.sum()
        mean += acc
    return mean / len(y_true)


def recall(y_true, y_pred):
    mean = 0
    for y, z in zip(y_true, y_pred):
        if z.sum() == 0:
            continue
        acc = (y * z).sum() / z.sum()
        mean += acc
    return mean / len(y_true)

def IoU(y_true, y_pred):
    from numpy import maximum
    mean = 0
    for y, z in zip(y_true, y_pred):
        if y.sum() + z.sum() == 0:
            continue
        intersection = (y * z)
        acc = intersection.sum() / maximum(y, z).sum()
        mean += acc
    return mean / len(y_true)