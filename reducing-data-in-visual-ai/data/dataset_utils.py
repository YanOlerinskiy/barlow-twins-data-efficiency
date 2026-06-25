import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_config import tiny_imagenet_transform


class ViTDatasetWrapper(Dataset):
    """
    A custom PyTorch Dataset wrapper for Hugging Face data.
    ViT models expect a dictionary with the key 'pixel_values'.
    """
    
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return {
            'pixel_values': image,
            'label': item['label']
        }

def get_pretraining_dataloader(split: int, batch_size=64, num_workers=4) -> ViTDatasetWrapper:
    """
    Loads the requested data split and returns a ready-to-use DataLoader.
    """

    raw_dataset = load_dataset("Maysee/tiny-imagenet", split="train")
    transformed_dataset = ViTDatasetWrapper(raw_dataset, transform=tiny_imagenet_transform)
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "tiny_imagenet_shuffled_indices.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing JSON split file at {json_path}. Run generate_splits.py first.")
        
    with open(json_path, "r") as f:
        shuffled_indices = json.load(f)
    
    split_indices = shuffled_indices[:split]
        
    subset = Subset(transformed_dataset, split_indices)
    
    loader = DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader