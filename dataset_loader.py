import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as albu

class DefectDataset(Dataset):
    
    def __init__(self, images_dir, masks_dir, ids=None, augmentation=None, preprocessing=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        #use the 'b' files as the unique identifiers for each layer
        #sort files to ensure alignment
        if ids:
            self.ids = ids
        else:
            self.ids = [f for f in os.listdir(images_dir) if f.endswith('b.PNG')]
        self.ids.sort()
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        #identifier for this layer
        filename_b = self.ids[i]
        base_name = filename_b.replace('b.PNG', '')
        
        #construct paths for the a,b,c
        path_a = os.path.join(self.images_dir, base_name + 'a.PNG')
        path_b = os.path.join(self.images_dir, filename_b)
        path_c = os.path.join(self.images_dir, base_name + 'c.PNG')
        
        #load all 3 images as grayscale (Mode 'L')
        img_a = np.array(Image.open(path_a).convert('L'))
        img_b = np.array(Image.open(path_b).convert('L'))
        img_c = np.array(Image.open(path_c).convert('L'))
        
        #stack into a 3-channel image
        #shape becomes (height, width, 3)
        image = np.dstack((img_a, img_b, img_c))
        
        #load the mask
        mask = Image.open(os.path.join(self.masks_dir, filename_b))
        mask = np.array(mask)
        
        #binarize mask (0.0 = background, 1.0 = defect)
        mask = np.where(mask > 0, 1.0, 0.0).astype('float32')
        
        #apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        #apply preprocessing (normalization for MobileNet)
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
            #if image (H, W, C) -> Transpose to (C, H, W)
            return x.transpose(2, 0, 1).astype('float32')
        else:
            #if mask (H, W) -> Add channel dim -> (H, W, 1) -> Transpose to (1, H, W)
            return np.expand_dims(x, axis=2).transpose(2, 0, 1).astype('float32')

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)