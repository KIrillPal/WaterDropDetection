class ListRandomCrop:
    """Crop the given PIL.Images at the same random location.
    Args:
        size (sequence or int): Desired output size of the crop.
    """
    image_crop_position = {}

    def __init__(self, size):
        import random
        self.random = random
        self.size = size

    def __call__(self, images, same_size=True):
        """
        Args:
            images: list<PIL.Image>: Images to be cropped.
            same_size: bool: Flag if all images have the same size.
        Returns:
            list<PIL.Image>: Cropped images.
        """
        if not images:
            return

        if same_size:
            w, h = images[0].size
        else:
            w = min([img.width for img in images])
            h = min([img.height for img in images])
        crop_h, crop_w = self.size

        x1 = self.random.randint(0, w - crop_w)
        y1 = self.random.randint(0, h - crop_h)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        return [img.crop((x1, y1, x2, y2)) for img in images]