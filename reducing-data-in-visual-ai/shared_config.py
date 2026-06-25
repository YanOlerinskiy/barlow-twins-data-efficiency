from torchvision import transforms

VIT_TINY_8_KWARGS = {
    "image_size": 64,
    "patch_size": 8,
    "num_channels": 3,
    "hidden_size": 192,
    "num_hidden_layers": 12,
    "num_attention_heads": 3,
    "intermediate_size": 768
}

tiny_imagenet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])