import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu

class DefectDataset(BaseDataset):
    
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        #sort files to ensure alignment
        self.ids.sort()
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
       
        image = Image.open(self.images_fps[i]).convert('RGB')
        image = np.array(image)
        
        mask = Image.open(self.masks_fps[i])
        mask = np.array(mask)
        
        mask = np.where(mask > 0, 1.0, 0.0).astype('float32')
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0.1, 0.1), p=0.5),
    ]
    return albu.Compose(train_transform)

def get_preprocessing(preprocessing_fn):
    
    def to_tensor(x, **kwargs):
        """
        Convert image or mask to PyTorch tensor.
        Handles both 3D images (H, W, C) and 2D masks (H, W).
        """
        if len(x.shape) == 3:
            # If image (H, W, C) -> Transpose to (C, H, W)
            return x.transpose(2, 0, 1).astype('float32')
        else:
            # If mask (H, W) -> Add channel dim -> (H, W, 1) -> Transpose to (1, H, W)
            return np.expand_dims(x, axis=2).transpose(2, 0, 1).astype('float32')

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)