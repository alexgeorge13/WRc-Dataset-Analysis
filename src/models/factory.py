import os
import timm
import torch
import torch.nn as nn
import torchvision.models as models

# Points to a hidden directory in your project root (ignored by your .gitignore)
LOCAL_CACHE_DIR = ".cache/model_weights/"

def get_cached_backbone(model_name, is_timm=True):
    """Checks local system cache for backbone weights; fetches once from hub if missing."""
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    save_path = os.path.join(LOCAL_CACHE_DIR, f"{model_name}.pth")

    if is_timm:
        if os.path.exists(save_path):
            print(f"📦 Loading {model_name} from local cache...")
            m = timm.create_model(model_name, pretrained=False, num_classes=0)
            m.load_state_dict(torch.load(save_path, map_location='cpu'))
        else:
            print(f"🌐 Cache miss. Downloading {model_name} from Hub...")
            m = timm.create_model(model_name, pretrained=True, num_classes=0)
            torch.save(m.state_dict(), save_path)
    else:
        # Handling Torchvision ResNet50
        if os.path.exists(save_path):
            print(f"📦 Loading ResNet50 from local cache...")
            m = models.resnet50(weights=None)
            m.load_state_dict(torch.load(save_path, map_location='cpu'))
        else:
            print(f"🌐 Cache miss. Downloading ResNet50 from Torchvision...")
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            torch.save(m.state_dict(), save_path)
            
    return m

def get_model(name, num_classes, device='cuda'):
    """Unified factory selector for Offboard Server and Onboard Edge architectures."""
    
    # ==========================================
    # TIER 1: OFFBOARD HIGH-CAPACITY SERVERS (~22M - 30M params)
    # ==========================================
    if name == "resnet50_standard":
        backbone = get_cached_backbone("resnet50", is_timm=False)
        backbone.fc = nn.Linear(2048, num_classes)
        return backbone.to(device)

    elif name == "swin_standard":
        backbone = get_cached_backbone('swin_tiny_patch4_window7_224', is_timm=True)
        model = nn.Sequential(backbone, nn.Linear(768, num_classes))
        return model.to(device)

    elif name == "vit_small":
        backbone = get_cached_backbone('vit_small_patch16_224', is_timm=True)
        model = nn.Sequential(backbone, nn.Linear(384, num_classes))
        return model.to(device)

    # ==========================================
    # TIER 2: ONBOARD LIGHTWEIGHT EDGE DEVICES (~3.5M - 5.7M params)
    # ==========================================
    elif name == "vit_tiny":
        backbone = get_cached_backbone('vit_tiny_patch16_224', is_timm=True)
        model = nn.Sequential(backbone, nn.Linear(192, num_classes))
        return model.to(device)
        
    elif name == "mobilenetv4_edge":
        # Corrected valid identifier tag & mapped 1280 pre-logits hidden features
        backbone = get_cached_backbone('mobilenetv4_conv_small.e2400_r224_in1k', is_timm=True)
        model = nn.Sequential(backbone, nn.Linear(1280, num_classes))
        return model.to(device)
        
    else:
        raise ValueError(f"Unknown model selector config: {name}")