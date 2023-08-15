# File with main measures for binary segmentation
# Working with ndarrays

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
        acc = (y * z).sum() / y.sum()
        mean += acc
    return mean / len(y_true)


def recall(y_true, y_pred):
    mean = 0
    for y, z in zip(y_true, y_pred):
        acc = (y * z).sum() / z.sum()
        mean += acc
    return mean / len(y_true)