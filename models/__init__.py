from .clip_models import CLIPModel
from .cie_iapl import CIEIAPLModel

VALID_NAMES = [
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
]

def build_model(args):
    if not args.backbone.startswith("CLIP:"):
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    assert args.backbone in VALID_NAMES

    model_variant = getattr(args, "model_variant", "clip_adapter")
    if model_variant == "clip_adapter":
        return CLIPModel(args)
    elif model_variant == "cie_iapl":
        return CIEIAPLModel(args)
    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")
