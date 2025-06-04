import numpy as np
from PIL import Image
import os
import sys
import torch
from huggingface_hub import snapshot_download
import gradio as gr

# Add Change-Clothes-AI to path
sys.path.append('/workspace/Change-Clothes-AI')

# Import Leffa components
from leffa.transform import LeffaTransform
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Import enhanced components
# Note: These files need to be created separately with the content from our previous steps
try:
    from leffa_utils.unified_mask_generator import UnifiedMaskGenerator
    from leffa_utils.model_adapter import ModelRegistry
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    print("Enhanced components not available. Using standard Leffa components.")
    ENHANCED_COMPONENTS_AVAILABLE = False

# Download checkpoints if needed
if not os.path.exists("./ckpts"):
    snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")

class EnhancedLeffaPredictor:
    """Enhanced Leffa predictor that integrates with Change-Clothes-AI components."""
    
    def __init__(self):
        """Initialize the predictor with components from both projects."""
        # Standard components initialization
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )
        
        # Initialize enhanced components if available
        if ENHANCED_COMPONENTS_AVAILABLE:
            self.unified_mask_generator = UnifiedMaskGenerator(
                mode="auto",
                densepose_path="./ckpts/densepose",
                schp_path="./ckpts/schp",
            )
            
            # Initialize model registry with all available models
            available_models = ModelRegistry.list_available_models()
            self.available_models = available_models
            
            # Load Leffa models by default
            self.vt_model_hd = ModelRegistry.get_model(
                model_type="virtual_tryon", 
                model_name="leffa_vt_hd"
            ).load()
            
            self.vt_model_dc = ModelRegistry.get_model(
                model_type="virtual_tryon", 
                model_name="leffa_vt_dc"
            ).load()
            
            self.pt_model = ModelRegistry.get_model(
                model_type="pose_transfer", 
                model_name="leffa_pt"
            ).load()
            
            # Load Change-Clothes-AI model if available
            if "change_clothes_vt" in available_models["virtual_tryon"]:
                self.cc_model = ModelRegistry.get_model(
                    model_type="virtual_tryon", 
                    model_name="change_clothes_vt"
                )
                # Not loading by default to save memory
                self.has_cc_model = True
            else:
                self.has_cc_model = False
        else:
            # Fall back to standard Leffa models
            from leffa.model import LeffaModel
            from leffa.inference import LeffaInference
            
            vt_model_hd = LeffaModel(
                pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
                pretrained_model="./ckpts/virtual_tryon.pth",
                dtype="float16",
            )
            self.vt_inference_hd = LeffaInference(model=vt_model_hd)

            vt_model_dc = LeffaModel(
                pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
                pretrained_model="./ckpts/virtual_tryon_dc.pth",
                dtype="float16",
            )
            self.vt_inference_dc = LeffaInference(model=vt_model_dc)

            pt_model = LeffaModel(
                pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
                pretrained_model="./ckpts/pose_transfer.pth",
                dtype="float16",
            )
            self.pt_inference = LeffaInference(model=pt_model)
            
            self.has_cc_model = False
    
    def enhanced_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        mask_mode="auto",
        model_name=None,  # If None, use default Leffa model
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        preprocess_garment=False,
        prompt=None,
        negative_prompt=None
    ):
        """
        Enhanced prediction with support for both Leffa and Change-Clothes-AI.
        
        Args:
            src_image_path: Path to source image
            ref_image_path: Path to reference image
            control_type: Type of control - "virtual_tryon" or "pose_transfer"
            mask_mode: Mask generation mode - "leffa", "change_clothes", "auto", or "hybrid"
            model_name: Name of the model to use - if None, use default Leffa model
            ref_acceleration: Whether to use reference acceleration
            step: Number of inference steps
            scale: Guidance scale
            seed: Random seed
            vt_model_type: Virtual try-on model type - "viton_hd" or "dress_code"
            vt_garment_type: Garment type - "upper_body", "lower_body", or "dresses"
            vt_repaint: Whether to use repaint
            preprocess_garment: Whether to preprocess the garment image
            prompt: Text prompt (for Change-Clothes-AI models)
            negative_prompt: Negative text prompt (for Change-Clothes-AI models)
            
        Returns:
            tuple: (generated_image, mask, densepose)
        """
        # Open and resize the source image
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, 768, 1024)

        # For virtual try-on, optionally preprocess the garment image
        if control_type == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            ref_image = Image.open(ref_image_path)
            
        ref_image = resize_and_center(ref_image, 768, 1024)

        # Process with enhanced components if available
        if ENHANCED_COMPONENTS_AVAILABLE:
            # Set mask generator mode
            self.unified_mask_generator.set_mode(mask_mode)
            
            # Generate parser result and keypoints
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            
            # Generate mask
            mask, mask_gray = self.unified_mask_generator.generate_mask(
                human_image=src_image,
                parser_result=model_parse,
                keypoints=keypoints,
                category=vt_garment_type,
                model_type=vt_model_type,
                return_gray=True
            )
            
            # Generate densepose
            src_image_array = np.array(src_image)
            
            if control_type == "virtual_tryon":
                if vt_model_type == "viton_hd":
                    src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                    densepose = Image.fromarray(src_image_seg_array)
                elif vt_model_type == "dress_code":
                    src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                    src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                    src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
                    densepose = Image.fromarray(src_image_seg_array)
            elif control_type == "pose_transfer":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
                densepose = Image.fromarray(src_image_iuv_array)
            
            # Select model based on provided parameters
            if model_name is not None and model_name.startswith("change_clothes") and self.has_cc_model:
                # Use Change-Clothes-AI model
                self.cc_model.load()
                output = self.cc_model.infer(
                    src_image=src_image,
                    ref_image=ref_image,
                    mask=mask,
                    densepose=densepose,
                    prompt=prompt or "a person wearing clothes",
                    negative_prompt=negative_prompt or "low quality, bad quality",
                    num_inference_steps=step,
                    guidance_scale=scale,
                    seed=seed
                )
                self.cc_model.unload()  # Unload to save memory
            else:
                # Use appropriate Leffa model
                if control_type == "virtual_tryon":
                    if vt_model_type == "viton_hd":
                        model = self.vt_model_hd
                    else:
                        model = self.vt_model_dc
                else:
                    model = self.pt_model
                
                output = model.infer(
                    src_image=src_image,
                    ref_image=ref_image,
                    mask=mask,
                    densepose=densepose,
                    ref_acceleration=ref_acceleration,
                    num_inference_steps=step,
                    guidance_scale=scale,
                    seed=seed,
                    repaint=vt_repaint
                )
            
            return np.array(output["generated_image"][0]), np.array(mask), np.array(densepose)
        else:
            # Fall back to standard Leffa processing
            src_image_array = np.array(src_image)

            if control_type == "virtual_tryon":
                src_image = src_image.convert("RGB")
                model_parse, _ = self.parsing(src_image.resize((384, 512)))
                keypoints = self.openpose(src_image.resize((384, 512)))
                if vt_model_type == "viton_hd":
                    mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
                elif vt_model_type == "dress_code":
                    mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
                mask = mask.resize((768, 1024))
            elif control_type == "pose_transfer":
                mask = Image.fromarray(np.ones_like(src_image_array) * 255)

            if control_type == "virtual_tryon":
                if vt_model_type == "viton_hd":
                    src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                    src_image_seg = Image.fromarray(src_image_seg_array)
                    densepose = src_image_seg
                elif vt_model_type == "dress_code":
                    src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                    src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                    src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
                    src_image_seg = Image.fromarray(src_image_seg_array)
                    densepose = src_image_seg
            elif control_type == "pose_transfer":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
                src_image_iuv = Image.fromarray(src_image_iuv_array)
                densepose = src_image_iuv

            transform = LeffaTransform()
            data = {
                "src_image": [src_image],
                "ref_image": [ref_image],
                "mask": [mask],
                "densepose": [densepose],
            }
            data = transform(data)
            if control_type == "virtual_tryon":
                if vt_model_type == "viton_hd":
                    inference = self.vt_inference_hd
                elif vt_model_type == "dress_code":
                    inference = self.vt_inference_dc
            elif control_type == "pose_transfer":
                inference = self.pt_inference
            output = inference(
                data,
                ref_acceleration=ref_acceleration,
                num_inference_steps=step,
                guidance_scale=scale,
                seed=seed,
                repaint=vt_repaint,
            )
            gen_image = output["generated_image"][0]
            return np.array(gen_image), np.array(mask), np.array(densepose)

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint, preprocess_garment, mask_mode="auto", model_name=None):
        return self.enhanced_predict(
            src_image_path,
            ref_image_path,
            "virtual_tryon",
            mask_mode,
            model_name,
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            vt_repaint,
            preprocess_garment,
        )

    def leffa_predict_pt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, mask_mode="auto", model_name=None):
        return self.enhanced_predict(
            src_image_path,
            ref_image_path,
            "pose_transfer",
            mask_mode,
            model_name,
            ref_acceleration,
            step,
            scale,
            seed,
        )


if __name__ == "__main__":
    leffa_predictor = EnhancedLeffaPredictor()
    example_dir = "./ckpts/examples"
    person1_images = list_dir(f"{example_dir}/person1")
    person2_images = list_dir(f"{example_dir}/person2")
    garment_images = list_dir(f"{example_dir}/garment")

    title = "## Enhanced Leffa: Combining Leffa and Change-Clothes-AI"
    link = """[📚 Based on Leffa](https://github.com/franciszzj/Leffa) and [Change-Clothes-AI](https://github.com/your-username/Change-Clothes-AI)
           
           This enhanced version combines techniques from both projects for better results.
           """
    description = "Enhanced Leffa is a unified framework for controllable person image generation that enables precise manipulation of both appearance (i.e., virtual try-on) and pose (i.e., pose transfer) by integrating techniques from both Leffa and Change-Clothes-AI."
    
    # Determine available models
    available_models = ["leffa_default"]
    available_mask_modes = ["auto", "leffa", "change_clothes", "hybrid"]
    
    if ENHANCED_COMPONENTS_AVAILABLE and hasattr(leffa_predictor, "has_cc_model") and leffa_predictor.has_cc_model:
        available_models.append("change_clothes")

    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink, secondary_hue=gr.themes.colors.red)).queue() as demo:
        gr.Markdown(title)
        gr.Markdown(link)
        gr.Markdown(description)

        with gr.Tab("Control Appearance (Virtual Try-on)"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        vt_src_image = gr.Image(type="filepath", label="Source Image")
                        vt_ref_image = gr.Image(type="filepath", label="Reference Garment")
                    
                    with gr.Row():
                        vt_model_dropdown = gr.Dropdown(
                            choices=available_models, 
                            value="leffa_default", 
                            label="Model"
                        )
                        vt_mask_mode_dropdown = gr.Dropdown(
                            choices=available_mask_modes, 
                            value="auto", 
                            label="Mask Generation Mode"
                        )
                    
                    with gr.Row():
                        vt_model_type = gr.Radio(
                            choices=["viton_hd", "dress_code"], 
                            value="viton_hd", 
                            label="Model Type"
                        )
                        vt_garment_type = gr.Radio(
                            choices=["upper_body", "lower_body", "dresses"], 
                            value="upper_body", 
                            label="Garment Type"
                        )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        with gr.Row():
                            vt_ref_acceleration = gr.Checkbox(value=False, label="Reference Acceleration")
                            vt_preprocess_garment = gr.Checkbox(value=False, label="Preprocess Garment")
                            vt_repaint = gr.Checkbox(value=False, label="Repaint")
                        
                        with gr.Row():
                            vt_step = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Step")
                            vt_scale = gr.Slider(minimum=1, maximum=10, value=2.5, step=0.1, label="Scale")
                            vt_seed = gr.Number(value=42, precision=0, label="Seed")
                        
                        with gr.Row():
                            vt_prompt = gr.Textbox(
                                label="Prompt (Change-Clothes-AI model only)", 
                                placeholder="a person wearing clothes",
                                visible=False,
                                interactive=True
                            )
                            vt_negative_prompt = gr.Textbox(
                                label="Negative Prompt (Change-Clothes-AI model only)", 
                                placeholder="low quality, bad quality",
                                visible=False,
                                interactive=True
                            )
                    
                    vt_submit = gr.Button("Submit")
                
                with gr.Column():
                    vt_output = gr.Image(label="Output")
                    
                    with gr.Accordion("Intermediate Results", open=False):
                        with gr.Row():
                            vt_mask = gr.Image(label="Mask")
                            vt_densepose = gr.Image(label="Densepose")
            
            vt_submit.click(
                fn=leffa_predictor.leffa_predict_vt,
                inputs=[
                    vt_src_image, vt_ref_image, 
                    vt_ref_acceleration, vt_step, vt_scale, vt_seed, 
                    vt_model_type, vt_garment_type, vt_repaint, vt_preprocess_garment,
                    vt_mask_mode_dropdown, vt_model_dropdown
                ],
                outputs=[vt_output, vt_mask, vt_densepose],
            )
            
            # Show/hide prompt inputs based on model selection
            def toggle_prompt_visibility(model_name):
                return {"visible": model_name == "change_clothes"}
                
            vt_model_dropdown.change(
                fn=toggle_prompt_visibility,
                inputs=[vt_model_dropdown],
                outputs=[vt_prompt],
            )
            
            vt_model_dropdown.change(
                fn=toggle_prompt_visibility,
                inputs=[vt_model_dropdown],
                outputs=[vt_negative_prompt],
            )
            
            # Setup examples
            vt_examples = []
            for i in range(min(2, len(person1_images))):
                for j in range(min(3, len(garment_images))):
                    vt_examples.append([
                        person1_images[i], garment_images[j], 
                        False, 50, 2.5, 42, 
                        "viton_hd", "upper_body", False, False, "auto", "leffa_default"
                    ])
            
            gr.Examples(
                examples=vt_examples,
                inputs=[
                    vt_src_image, vt_ref_image, 
                    vt_ref_acceleration, vt_step, vt_scale, vt_seed, 
                    vt_model_type, vt_garment_type, vt_repaint, vt_preprocess_garment,
                    vt_mask_mode_dropdown, vt_model_dropdown
                ],
                outputs=[vt_output, vt_mask, vt_densepose],
                fn=leffa_predictor.leffa_predict_vt,
                cache_examples=True,
            )

        with gr.Tab("Control Pose (Pose Transfer)"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        pt_src_image = gr.Image(type="filepath", label="Source Image")
                        pt_ref_image = gr.Image(type="filepath", label="Reference Pose")
                    
                    with gr.Row():
                        pt_mask_mode_dropdown = gr.Dropdown(
                            choices=available_mask_modes, 
                            value="auto", 
                            label="Mask Generation Mode"
                        )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        with gr.Row():
                            pt_ref_acceleration = gr.Checkbox(value=False, label="Reference Acceleration")
                        
                        with gr.Row():
                            pt_step = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Step")
                            pt_scale = gr.Slider(minimum=1, maximum=10, value=2.5, step=0.1, label="Scale")
                            pt_seed = gr.Number(value=42, precision=0, label="Seed")
                    
                    pt_submit = gr.Button("Submit")
                
                with gr.Column():
                    pt_output = gr.Image(label="Output")
                    
                    with gr.Accordion("Intermediate Results", open=False):
                        with gr.Row():
                            pt_mask = gr.Image(label="Mask")
                            pt_densepose = gr.Image(label="Densepose")
            
            pt_submit.click(
                fn=leffa_predictor.leffa_predict_pt,
                inputs=[
                    pt_src_image, pt_ref_image, 
                    pt_ref_acceleration, pt_step, pt_scale, pt_seed,
                    pt_mask_mode_dropdown, gr.Textbox(value="leffa_default", visible=False)
                ],
                outputs=[pt_output, pt_mask, pt_densepose],
            )
            
            # Setup examples
            pt_examples = []
            for i in range(min(2, len(person1_images))):
                for j in range(min(2, len(person2_images))):
                    pt_examples.append([
                        person1_images[i], person2_images[j], 
                        False, 50, 2.5, 42, "auto", "leffa_default"
                    ])
            
            gr.Examples(
                examples=pt_examples,
                inputs=[
                    pt_src_image, pt_ref_image, 
                    pt_ref_acceleration, pt_step, pt_scale, pt_seed,
                    pt_mask_mode_dropdown, gr.Textbox(value="leffa_default", visible=False)
                ],
                outputs=[pt_output, pt_mask, pt_densepose],
                fn=leffa_predictor.leffa_predict_pt,
                cache_examples=True,
            )

    demo.launch(allowed_paths=[example_dir]) 