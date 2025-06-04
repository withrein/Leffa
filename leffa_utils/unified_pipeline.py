import os
import sys
from typing import Dict, List, Optional, Union, Tuple, Any
import torch
from PIL import Image
import numpy as np

# Add paths
sys.path.append('/workspace/Change-Clothes-AI')

# Import unified components
from leffa_utils.unified_mask_generator import UnifiedMaskGenerator
from leffa_utils.model_adapter import ModelRegistry, BaseModelAdapter

# Import preprocessing components
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, preprocess_garment_image


class UnifiedPipeline:
    """
    Unified pipeline that combines components from Leffa and Change-Clothes-AI.
    This pipeline provides a consistent interface for different combinations
    of preprocessing and model components.
    """
    
    def __init__(self, config=None):
        """
        Initialize the unified pipeline.
        
        Args:
            config (dict, optional): Configuration for the pipeline.
                If None, default configuration will be used.
        """
        # Set default config if not provided
        if config is None:
            config = {
                "mask_generator": {
                    "mode": "auto",
                    "densepose_path": "./ckpts/densepose",
                    "schp_path": "./ckpts/schp",
                },
                "model": {
                    "type": "virtual_tryon",
                    "name": "leffa_vt_hd",
                    "device": "cuda",
                },
                "preprocessing": {
                    "densepose": {
                        "config_path": "./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
                        "weights_path": "./ckpts/densepose/model_final_162be9.pkl",
                    },
                    "humanparsing": {
                        "atr_path": "./ckpts/humanparsing/parsing_atr.onnx",
                        "lip_path": "./ckpts/humanparsing/parsing_lip.onnx",
                    },
                    "openpose": {
                        "body_model_path": "./ckpts/openpose/body_pose_model.pth",
                    },
                },
            }
        self.config = config
        
        # Initialize components
        self._init_preprocessing()
        self._init_mask_generator()
        self._init_model()
        
    def _init_preprocessing(self):
        """Initialize preprocessing components."""
        preprocessing_config = self.config["preprocessing"]
        
        # Initialize DensePose predictor
        self.densepose_predictor = DensePosePredictor(
            config_path=preprocessing_config["densepose"]["config_path"],
            weights_path=preprocessing_config["densepose"]["weights_path"],
        )
        
        # Initialize human parsing
        self.parsing = Parsing(
            atr_path=preprocessing_config["humanparsing"]["atr_path"],
            lip_path=preprocessing_config["humanparsing"]["lip_path"],
        )
        
        # Initialize OpenPose
        self.openpose = OpenPose(
            body_model_path=preprocessing_config["openpose"]["body_model_path"],
        )
    
    def _init_mask_generator(self):
        """Initialize mask generator."""
        mask_config = self.config["mask_generator"]
        self.mask_generator = UnifiedMaskGenerator(
            mode=mask_config["mode"],
            densepose_path=mask_config["densepose_path"],
            schp_path=mask_config["schp_path"],
        )
    
    def _init_model(self):
        """Initialize model."""
        model_config = self.config["model"]
        self.model = ModelRegistry.get_model(
            model_type=model_config["type"],
            model_name=model_config["name"],
            device=model_config["device"],
        )
        
    def set_mask_generator_mode(self, mode):
        """
        Set the mask generator mode.
        
        Args:
            mode (str): The mode to use - "leffa", "change_clothes", "auto", or "hybrid"
        """
        self.mask_generator.set_mode(mode)
        self.config["mask_generator"]["mode"] = mode
        
    def set_model(self, model_type, model_name):
        """
        Set the model to use.
        
        Args:
            model_type (str): The model type - "virtual_tryon" or "pose_transfer"
            model_name (str): The specific model name
        """
        # Unload current model if it exists
        if hasattr(self, "model") and self.model is not None:
            self.model.unload()
            
        # Load new model
        self.model = ModelRegistry.get_model(
            model_type=model_type,
            model_name=model_name,
            device=self.config["model"]["device"],
        )
        
        # Update config
        self.config["model"]["type"] = model_type
        self.config["model"]["name"] = model_name
        
    def process(self, 
                src_image_path, 
                ref_image_path, 
                category="upper_body", 
                model_type="viton_hd",
                preprocess_garment=False,
                **kwargs):
        """
        Process images through the pipeline.
        
        Args:
            src_image_path (str): Path to the source image
            ref_image_path (str): Path to the reference image (garment or pose)
            category (str): The clothing category - "upper_body", "lower_body", "dresses"
            model_type (str): The model type - "viton_hd" or "dress_code"
            preprocess_garment (bool): Whether to preprocess the garment image
            **kwargs: Additional parameters for the model
                - ref_acceleration (bool): Whether to use reference acceleration
                - num_inference_steps (int): Number of inference steps
                - guidance_scale (float): Guidance scale
                - seed (int): Random seed
                - repaint (bool): Whether to use repaint
                
        Returns:
            dict: Results including the generated image, mask, and densepose
        """
        # 1. Load and preprocess images
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, 768, 1024)
        
        # For virtual try-on, optionally preprocess the garment (reference) image
        if self.config["model"]["type"] == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            ref_image = Image.open(ref_image_path)
            
        ref_image = resize_and_center(ref_image, 768, 1024)
        
        # 2. Generate parsing and keypoints
        model_parse, _ = self.parsing(src_image.resize((384, 512)))
        keypoints = self.openpose(src_image.resize((384, 512)))
        
        # 3. Generate mask
        mask, mask_gray = self.mask_generator.generate_mask(
            human_image=src_image,
            parser_result=model_parse,
            keypoints=keypoints,
            category=category,
            model_type=model_type,
            return_gray=True
        )
        
        # 4. Generate densepose
        src_image_array = np.array(src_image)
        
        if self.config["model"]["type"] == "virtual_tryon":
            if model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                densepose = Image.fromarray(src_image_seg_array)
            elif model_type == "dress_code":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
                densepose = Image.fromarray(src_image_seg_array)
        elif self.config["model"]["type"] == "pose_transfer":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
            densepose = Image.fromarray(src_image_iuv_array)
        
        # 5. Run inference
        output = self.model.infer(
            src_image=src_image,
            ref_image=ref_image,
            mask=mask,
            densepose=densepose,
            **kwargs
        )
        
        # 6. Add intermediate results to output
        output["mask"] = mask
        output["mask_gray"] = mask_gray
        output["densepose"] = densepose
        
        return output
    
    @staticmethod
    def list_available_models():
        """List all available models."""
        return ModelRegistry.list_available_models()
    
    @staticmethod
    def list_available_mask_modes():
        """List all available mask generator modes."""
        return ["leffa", "change_clothes", "auto", "hybrid"]
    
    @staticmethod
    def list_available_categories():
        """List all available clothing categories."""
        return ["upper_body", "lower_body", "dresses"]
    
    @staticmethod
    def list_available_model_types():
        """List all available model types."""
        return ["viton_hd", "dress_code"]


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize pipeline
    pipeline = UnifiedPipeline()
    
    # List available models
    print("Available models:")
    for model_type, models in pipeline.list_available_models().items():
        print(f"  {model_type}: {', '.join(models)}")
    
    # Load test images if they exist
    test_src_path = "./ckpts/examples/person1/1.jpg"
    test_ref_path = "./ckpts/examples/garment/1.jpg"
    
    if os.path.exists(test_src_path) and os.path.exists(test_ref_path):
        # Process with different mask generator modes
        for mode in pipeline.list_available_mask_modes():
            pipeline.set_mask_generator_mode(mode)
            
            output = pipeline.process(
                src_image_path=test_src_path,
                ref_image_path=test_ref_path,
                category="upper_body",
                model_type="viton_hd",
                num_inference_steps=10,  # Low for quick testing
                seed=42
            )
            
            # Save results
            os.makedirs("./test_pipeline", exist_ok=True)
            output["generated_image"][0].save(f"./test_pipeline/generated_{mode}.png")
            output["mask"].save(f"./test_pipeline/mask_{mode}.png")
            output["mask_gray"].save(f"./test_pipeline/mask_gray_{mode}.png")
            output["densepose"].save(f"./test_pipeline/densepose_{mode}.png")
            
        print("Test results saved to ./test_pipeline/")
    else:
        print("Test images not found.") 