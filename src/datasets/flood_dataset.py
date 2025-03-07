import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import natsort

class FloodDataset(Dataset):
    """
    Dataset for flood segmentation.
    Just a normal dataset class with some additional methods for L-CRP:
        - class_names;
        - reverse_augmentation and reverse_normalization. - them I took from coco2017 dataset, I'm not sure if they are correct for flood dataset, but looks like they work.
    """
    class_names = ["background", "flood"]

    def __init__(self, root_dir, split="train", transform=None, image_size=(480, 480)):
        self.image_dir = os.path.join(root_dir, "RGB", split, "JPEG")
        self.mask_dir = os.path.join(root_dir, "annotations", split, "JPEG")
        self.transform = transform
        self.image_size = image_size  # Target size for resizing

        # Get list of image and mask files, ensuring they are sorted correctly
        self.image_files = natsort.natsorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = natsort.natsorted([f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Ensure the number of images and masks match
        assert len(self.image_files) == len(self.mask_files), "Mismatch between number of images and masks!"

        # Define resize transform
        self.resize = transforms.Resize(self.image_size, transforms.InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize(self.image_size, transforms.InterpolationMode.NEAREST)  # Nearest for masks
        self.reverse_normalization = torch.nn.Identity()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")  # Load RGB image
        mask = Image.open(mask_path).convert("L")  # Load mask in grayscale

        # Resize images and masks
        image = self.resize(image)
        mask = self.resize_mask(mask)

        # Apply additional transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    
    def reverse_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        return torch.multiply(data, 255).type(torch.uint8).detach().cpu()
    