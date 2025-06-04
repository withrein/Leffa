import os
import sys
from typing import Dict, List, Optional, Union, Tuple, Any
import torch
from PIL import Image
import numpy as np
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)

# Add Change-Clothes-AI to path
sys.path.append('/workspace/Change-Clothes-AI')

# Import Leffa components
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa.transform import LeffaTransform

# Import Change-Clothes-AI components
try:
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel
    CHANGE_CLOTHES_AVAILABLE = True
except ImportError:
    print("Warning: Change-Clothes-AI components not found. Only Leffa models will be available.")
    CHANGE_CLOTHES_AVAILABLE = False

class ModelRegistry:
    """Registry for different model types with consistent interface."""
    
    @staticmethod
    def list_available_models():
        """List all available models."""
        models = {
            "virtual_tryon": ["leffa_vt_hd", "leffa_vt_dc"],
            "pose_transfer": ["leffa_pt"]
        }
        
        if CHANGE_CLOTHES_AVAILABLE:
            models["virtual_tryon"].append("change_clothes_vt")
            
        return models
    
    @staticmethod
    def get_model(model_type, model_name, device="cuda"):
        """
        Get a model instance by type and name.
        
        Args:
            model_type (str): The model type - "virtual_tryon" or "pose_transfer"
            model_name (str): The specific model name
            device (str): The device to load the model on
            
        Returns:
            BaseModelAdapter: An instance of the model adapter
        """
        # Validate model type
        if model_type not in ["virtual_tryon", "pose_transfer"]:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Create and return the appropriate model
        if model_name == "leffa_vt_hd":
            return LeffaModelAdapter(
                model_type="virtual_tryon",
                pretrained_model_path="./ckpts/virtual_tryon.pth",
                base_model_path="./ckpts/stable-diffusion-inpainting",
                device=device
            )
        elif model_name == "leffa_vt_dc":
            return LeffaModelAdapter(
                model_type="virtual_tryon",
                pretrained_model_path="./ckpts/virtual_tryon_dc.pth",
                base_model_path="./ckpts/stable-diffusion-inpainting",
                device=device
            )
        elif model_name == "leffa_pt":
            return LeffaModelAdapter(
                model_type="pose_transfer",
                pretrained_model_path="./ckpts/pose_transfer.pth",
                base_model_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
                device=device
            )
        elif model_name == "change_clothes_vt" and CHANGE_CLOTHES_AVAILABLE:
            return ChangeClothesModelAdapter(
                base_path="/workspace/Change-Clothes-AI",
                model_path="yisol/IDM-VTON",
                device=device
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")


class BaseModelAdapter:
    """Base class for model adapters with a consistent interface."""
    
    def __init__(self, model_type, device="cuda"):
        """
        Initialize the model adapter.
        
        Args:
            model_type (str): The model type - "virtual_tryon" or "pose_transfer"
            device (str): The device to load the model on
        """
        self.model_type = model_type
        self.device = device
        
    def load(self):
        """Load the model. Should be implemented by subclasses."""
        raise NotImplementedError
        
    def unload(self):
        """Unload the model to free memory."""
        raise NotImplementedError
        
    def infer(self, src_image, ref_image, mask, densepose, **kwargs):
        """
        Run inference with the model.
        
        Args:
            src_image (PIL.Image): The source image
            ref_image (PIL.Image): The reference image (garment or pose)
            mask (PIL.Image): The mask
            densepose (PIL.Image): The densepose image
            **kwargs: Additional parameters for the model
            
        Returns:
            dict: Model outputs including the generated image
        """
        raise NotImplementedError


class LeffaModelAdapter(BaseModelAdapter):
    """Adapter for Leffa models."""
    
    def __init__(self, model_type, pretrained_model_path, base_model_path, device="cuda"):
        """
        Initialize the Leffa model adapter.
        
        Args:
            model_type (str): The model type - "virtual_tryon" or "pose_transfer"
            pretrained_model_path (str): Path to the pretrained model weights
            base_model_path (str): Path to the base diffusion model
            device (str): The device to load the model on
        """
        super().__init__(model_type, device)
        self.pretrained_model_path = pretrained_model_path
        self.base_model_path = base_model_path
        self.model = None
        self.inference = None
        self.transform = LeffaTransform()
        
    def load(self):
        """Load the Leffa model."""
        if self.model is None:
            print(f"Loading Leffa model from {self.pretrained_model_path}")
            self.model = LeffaModel(
                pretrained_model_name_or_path=self.base_model_path,
                pretrained_model=self.pretrained_model_path,
                dtype="float16",
            )
            self.inference = LeffaInference(model=self.model)
        return self
        
    def unload(self):
        """Unload the Leffa model."""
        if self.model is not None:
            del self.model
            del self.inference
            self.model = None
            self.inference = None
            torch.cuda.empty_cache()
        
    def infer(self, src_image, ref_image, mask, densepose, **kwargs):
        """
        Run inference with the Leffa model.
        
        Args:
            src_image (PIL.Image): The source image
            ref_image (PIL.Image): The reference image (garment or pose)
            mask (PIL.Image): The mask
            densepose (PIL.Image): The densepose image
            **kwargs: Additional parameters for the model
                - ref_acceleration (bool): Whether to use reference acceleration
                - num_inference_steps (int): Number of inference steps
                - guidance_scale (float): Guidance scale
                - seed (int): Random seed
                - repaint (bool): Whether to use repaint
            
        Returns:
            dict: Model outputs including the generated image
        """
        if self.inference is None:
            self.load()
            
        # Prepare data for Leffa
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = self.transform(data)
        
        # Extract parameters
        ref_acceleration = kwargs.get("ref_acceleration", False)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        guidance_scale = kwargs.get("guidance_scale", 2.5)
        seed = kwargs.get("seed", 42)
        repaint = kwargs.get("repaint", False)
        
        # Run inference
        output = self.inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            repaint=repaint,
        )
        
        return output


if CHANGE_CLOTHES_AVAILABLE:
    class ChangeClothesModelAdapter(BaseModelAdapter):
        """Adapter for Change-Clothes-AI models."""
        
        def __init__(self, base_path, model_path, device="cuda"):
            """
            Initialize the Change-Clothes-AI model adapter.
            
            Args:
                base_path (str): Path to the Change-Clothes-AI codebase
                model_path (str): Path or HF hub ID for the model
                device (str): The device to load the model on
            """
            super().__init__("virtual_tryon", device)
            self.base_path = base_path
            self.model_path = model_path
            self.pipe = None
            
        def load(self):
            """Load the Change-Clothes-AI model."""
            if self.pipe is None:
                print(f"Loading Change-Clothes-AI model from {self.model_path}")
                
                # Initialize model components
                unet = UNet2DConditionModel.from_pretrained(
                    self.model_path,
                    subfolder="unet",
                    torch_dtype=torch.float16,
                )
                unet.requires_grad_(False)
                
                tokenizer_one = AutoTokenizer.from_pretrained(
                    self.model_path,
                    subfolder="tokenizer",
                    revision=None,
                    use_fast=False,
                )
                
                tokenizer_two = AutoTokenizer.from_pretrained(
                    self.model_path,
                    subfolder="tokenizer_2",
                    revision=None,
                    use_fast=False,
                )
                
                noise_scheduler = DDPMScheduler.from_pretrained(
                    self.model_path, 
                    subfolder="scheduler"
                )
                
                text_encoder_one = CLIPTextModel.from_pretrained(
                    self.model_path,
                    subfolder="text_encoder",
                    torch_dtype=torch.float16,
                )
                
                text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                    self.model_path,
                    subfolder="text_encoder_2",
                    torch_dtype=torch.float16,
                )
                
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    self.model_path,
                    subfolder="image_encoder",
                    torch_dtype=torch.float16,
                )
                
                vae = AutoencoderKL.from_pretrained(
                    self.model_path,
                    subfolder="vae",
                    torch_dtype=torch.float16,
                )
                
                # Reference UNet encoder
                UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
                    self.model_path,
                    subfolder="unet_encoder",
                    torch_dtype=torch.float16,
                )
                
                # Ensure models don't require gradients
                UNet_Encoder.requires_grad_(False)
                image_encoder.requires_grad_(False)
                vae.requires_grad_(False)
                unet.requires_grad_(False)
                text_encoder_one.requires_grad_(False)
                text_encoder_two.requires_grad_(False)
                
                # Create pipeline
                self.pipe = TryonPipeline.from_pretrained(
                    self.model_path,
                    unet=unet,
                    vae=vae,
                    feature_extractor=CLIPImageProcessor(),
                    text_encoder=text_encoder_one,
                    text_encoder_2=text_encoder_two,
                    tokenizer=tokenizer_one,
                    tokenizer_2=tokenizer_two,
                    scheduler=noise_scheduler,
                    image_encoder=image_encoder,
                    torch_dtype=torch.float16,
                )
                self.pipe.unet_encoder = UNet_Encoder
                
                # Move to device
                self.pipe.to(self.device)
                self.pipe.unet_encoder.to(self.device)
                
            return self
            
        def unload(self):
            """Unload the Change-Clothes-AI model."""
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
                torch.cuda.empty_cache()
            
        def infer(self, src_image, ref_image, mask, densepose, **kwargs):
            """
            Run inference with the Change-Clothes-AI model.
            
            Args:
                src_image (PIL.Image): The source image
                ref_image (PIL.Image): The reference image (garment)
                mask (PIL.Image): The mask
                densepose (PIL.Image): The densepose image
                **kwargs: Additional parameters for the model
                    - prompt (str): Text prompt
                    - negative_prompt (str): Negative prompt
                    - num_inference_steps (int): Number of inference steps
                    - guidance_scale (float): Guidance scale
                    - seed (int): Random seed
                
            Returns:
                dict: Model outputs including the generated image
            """
            if self.pipe is None:
                self.load()
                
            # Extract parameters
            prompt = kwargs.get("prompt", "a person wearing clothes")
            negative_prompt = kwargs.get("negative_prompt", "low quality, bad quality")
            num_inference_steps = kwargs.get("num_inference_steps", 50)
            guidance_scale = kwargs.get("guidance_scale", 2.5)
            seed = kwargs.get("seed", 42)
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                
            # Convert mask for compatibility
            mask_gray = kwargs.get("mask_gray", None)
            
            # Run inference
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # Generate positive prompt embedding
                    pos_embed_dict = self.pipe._encode_prompt(
                        prompt=prompt,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    
                    # Generate images
                    generated_images = self.pipe(
                        prompt_embeds=pos_embed_dict['prompt_embeds'],
                        pooled_prompt_embeds=pos_embed_dict['pooled_prompt_embeds'],
                        negative_prompt_embeds=pos_embed_dict['negative_prompt_embeds'],
                        negative_pooled_prompt_embeds=pos_embed_dict['negative_pooled_prompt_embeds'],
                        reference_image=ref_image,
                        image=src_image,
                        mask_image=mask,
                        pose_img=densepose,
                        height=1024,
                        width=768,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                    ).images
            
            # Format output to match Leffa interface
            output = {
                "generated_image": generated_images,
                "src_image": [src_image],
                "ref_image": [ref_image],
                "mask": [mask]
            }
            
            return output


# Example usage
if __name__ == "__main__":
    import os
    from PIL import Image
    
    # List available models
    print("Available models:")
    for model_type, models in ModelRegistry.list_available_models().items():
        print(f"  {model_type}: {', '.join(models)}")
    
    # Test with a Leffa model
    model_adapter = ModelRegistry.get_model(
        model_type="virtual_tryon", 
        model_name="leffa_vt_hd"
    )
    
    # Load test images if they exist
    test_src_path = "./ckpts/examples/person1/1.jpg"
    test_ref_path = "./ckpts/examples/garment/1.jpg"
    test_mask_path = "./test_masks/mask_auto.png"  # From unified_mask_generator test
    test_densepose_path = "./ckpts/examples/person1/1_densepose.jpg"
    
    if (os.path.exists(test_src_path) and os.path.exists(test_ref_path) and 
            os.path.exists(test_mask_path) and os.path.exists(test_densepose_path)):
        
        src_image = Image.open(test_src_path).convert("RGB").resize((768, 1024))
        ref_image = Image.open(test_ref_path).convert("RGB").resize((768, 1024))
        mask = Image.open(test_mask_path).convert("L").resize((768, 1024))
        densepose = Image.open(test_densepose_path).convert("RGB").resize((768, 1024))
        
        # Test inference
        model_adapter.load()
        output = model_adapter.infer(
            src_image=src_image,
            ref_image=ref_image,
            mask=mask,
            densepose=densepose,
            num_inference_steps=10,  # Low for quick testing
            seed=42
        )
        
        # Save result
        os.makedirs("./test_results", exist_ok=True)
        output["generated_image"][0].save("./test_results/leffa_vt_hd_test.png")
        print("Test result saved to ./test_results/leffa_vt_hd_test.png")
        
        # Unload model
        model_adapter.unload()
        
        # Test with Change-Clothes-AI model if available
        if CHANGE_CLOTHES_AVAILABLE:
            model_adapter = ModelRegistry.get_model(
                model_type="virtual_tryon", 
                model_name="change_clothes_vt"
            )
            
            # Test inference
            model_adapter.load()
            output = model_adapter.infer(
                src_image=src_image,
                ref_image=ref_image,
                mask=mask,
                densepose=densepose,
                num_inference_steps=10,  # Low for quick testing
                seed=42
            )
            
            # Save result
            output["generated_image"][0].save("./test_results/change_clothes_vt_test.png")
            print("Test result saved to ./test_results/change_clothes_vt_test.png")
            
            # Unload model
            model_adapter.unload()
    else:
        print("Test images not found. Please run the unified_mask_generator test first.") 