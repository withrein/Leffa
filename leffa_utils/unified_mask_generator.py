import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Add Change-Clothes-AI to path to use its utilities
sys.path.append('/workspace/Change-Clothes-AI')

# Import Leffa utilities
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.utils import get_agnostic_mask_hd, get_agnostic_mask_dc

# Import Change-Clothes-AI utilities
from utils_mask import get_mask_location as change_clothes_get_mask

class UnifiedMaskGenerator:
    """Unified mask generator that combines approaches from Leffa and Change-Clothes-AI."""
    
    def __init__(self, mode="auto", 
                 densepose_path=None, 
                 schp_path=None):
        """
        Initialize the mask generator.
        
        Args:
            mode (str): The mode to use - "leffa", "change_clothes", "auto", or "hybrid"
            densepose_path (str): Path to densepose models
            schp_path (str): Path to SCHP models
        """
        self.mode = mode
        
        # Set default paths if not provided
        if densepose_path is None:
            densepose_path = "./ckpts/densepose"
        if schp_path is None:
            schp_path = "./ckpts/schp"
            
        # Initialize Leffa components if needed
        if self.mode in ["leffa", "auto", "hybrid"]:
            self.leffa_masker = AutoMasker(
                densepose_path=densepose_path,
                schp_path=schp_path,
            )
            
        # For Change-Clothes-AI, we don't need to initialize specific components
        # as it uses external parsing and pose models
        
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def generate_mask(self, human_image, parser_result, keypoints, 
                      category="upper_body", model_type="viton_hd",
                      return_gray=True):
        """
        Generate a mask for the given human image.
        
        Args:
            human_image (PIL.Image): The human image
            parser_result: The parser results from a human parsing model
            keypoints: The keypoints from an openpose model
            category (str): The clothing category - "upper_body", "lower_body", "dresses"
            model_type (str): The model type - "viton_hd" or "dress_code"
            return_gray (bool): Whether to return a grayscale mask as well
            
        Returns:
            tuple: (mask, gray_mask) if return_gray=True, otherwise just mask
        """
        # Ensure human_image is the right size
        if human_image.size != (768, 1024):
            human_image = human_image.resize((768, 1024))
            
        # Normalize category for consistency between projects
        normalized_category = self._normalize_category(category)
        
        if self.mode == "leffa":
            # Use Leffa's approach
            if model_type == "viton_hd":
                mask = get_agnostic_mask_hd(parser_result, keypoints, normalized_category)
            elif model_type == "dress_code":
                mask = get_agnostic_mask_dc(parser_result, keypoints, normalized_category)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            mask = mask.resize((768, 1024))
            
        elif self.mode == "change_clothes":
            # Use Change-Clothes-AI approach
            mode_prefix = "hd" if model_type == "viton_hd" else "dc"
            mask, gray_mask = change_clothes_get_mask(
                mode_prefix, normalized_category, parser_result, keypoints
            )
            mask = mask.resize((768, 1024))
            
        elif self.mode == "hybrid":
            # Combine both approaches - use Change-Clothes-AI for initial mask
            # then refine with Leffa's processing
            mode_prefix = "hd" if model_type == "viton_hd" else "dc"
            cc_mask, _ = change_clothes_get_mask(
                mode_prefix, normalized_category, parser_result, keypoints
            )
            cc_mask = cc_mask.resize((768, 1024))
            
            # Apply Leffa's additional processing
            if model_type == "viton_hd":
                leffa_mask = get_agnostic_mask_hd(parser_result, keypoints, normalized_category)
            else:
                leffa_mask = get_agnostic_mask_dc(parser_result, keypoints, normalized_category)
            leffa_mask = leffa_mask.resize((768, 1024))
            
            # Combine masks - use logical OR for better coverage
            cc_mask_np = np.array(cc_mask) > 127
            leffa_mask_np = np.array(leffa_mask) > 127
            combined_mask_np = np.logical_or(cc_mask_np, leffa_mask_np).astype(np.uint8) * 255
            mask = Image.fromarray(combined_mask_np)
            
        elif self.mode == "auto":
            # Try both approaches and use the one that gives better coverage
            # First, get Change-Clothes-AI mask
            mode_prefix = "hd" if model_type == "viton_hd" else "dc"
            cc_mask, _ = change_clothes_get_mask(
                mode_prefix, normalized_category, parser_result, keypoints
            )
            cc_mask = cc_mask.resize((768, 1024))
            
            # Then get Leffa mask
            if model_type == "viton_hd":
                leffa_mask = get_agnostic_mask_hd(parser_result, keypoints, normalized_category)
            else:
                leffa_mask = get_agnostic_mask_dc(parser_result, keypoints, normalized_category)
            leffa_mask = leffa_mask.resize((768, 1024))
            
            # Measure coverage and choose the better one
            cc_coverage = np.mean(np.array(cc_mask) > 127)
            leffa_coverage = np.mean(np.array(leffa_mask) > 127)
            
            # Choose the mask with better coverage, unless it's too much
            if 0.05 <= cc_coverage <= 0.4 and cc_coverage > leffa_coverage:
                mask = cc_mask
            elif 0.05 <= leffa_coverage <= 0.4:
                mask = leffa_mask
            else:
                # If both have poor coverage, combine them
                cc_mask_np = np.array(cc_mask) > 127
                leffa_mask_np = np.array(leffa_mask) > 127
                combined_mask_np = np.logical_or(cc_mask_np, leffa_mask_np).astype(np.uint8) * 255
                mask = Image.fromarray(combined_mask_np)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Generate grayscale mask if requested
        if return_gray:
            # Create grayscale mask that shows the human with the mask area zeroed out
            mask_tensor = 1 - transforms.ToTensor()(mask)
            human_tensor = self.tensor_transform(human_image)
            gray_mask = mask_tensor * human_tensor
            
            # Convert back to PIL
            gray_mask_pil = transforms.ToPILImage()((gray_mask + 1.0) / 2.0)
            return mask, gray_mask_pil
        
        return mask
    
    def _normalize_category(self, category):
        """Normalize category names between the two projects."""
        category = category.lower()
        
        if category in ["upper", "upper_body", "upper_clothes", "tops"]:
            return "upper_body"
        elif category in ["lower", "lower_body", "lower_clothes", "bottoms", "pants", "trousers"]:
            return "lower_body"
        elif category in ["dress", "dresses", "full", "full_body"]:
            return "dresses"
        else:
            return category  # Return as is if unknown

    def set_mode(self, mode):
        """Change the current mode."""
        valid_modes = ["leffa", "change_clothes", "auto", "hybrid"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        self.mode = mode


# Example usage
if __name__ == "__main__":
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    
    # Initialize models
    parsing_model = Parsing(
        atr_path="./ckpts/humanparsing/parsing_atr.onnx",
        lip_path="./ckpts/humanparsing/parsing_lip.onnx",
    )
    openpose_model = OpenPose(
        body_model_path="./ckpts/openpose/body_pose_model.pth",
    )
    
    # Initialize the unified mask generator
    mask_generator = UnifiedMaskGenerator(mode="auto")
    
    # Load a test image
    test_image_path = "./ckpts/examples/person1/1.jpg"
    if os.path.exists(test_image_path):
        human_image = Image.open(test_image_path).convert("RGB")
        human_image = human_image.resize((768, 1024))
        
        # Generate parser result and keypoints
        model_parse, _ = parsing_model(human_image.resize((384, 512)))
        keypoints = openpose_model(human_image.resize((384, 512)))
        
        # Generate masks with different modes
        for mode in ["leffa", "change_clothes", "hybrid", "auto"]:
            mask_generator.set_mode(mode)
            mask, gray_mask = mask_generator.generate_mask(
                human_image, model_parse, keypoints, 
                category="upper_body", model_type="viton_hd"
            )
            
            # Save the results
            os.makedirs("./test_masks", exist_ok=True)
            mask.save(f"./test_masks/mask_{mode}.png")
            gray_mask.save(f"./test_masks/gray_mask_{mode}.png")
            
        print("Test masks generated in ./test_masks/")
    else:
        print(f"Test image not found: {test_image_path}") 