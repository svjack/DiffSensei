import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr
import torch
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor, ViTImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModelWithProjection, AutoImageProcessor, AutoModel, LlamaTokenizer, ViTMAEModel
import numpy as np

from src.models.unet import UNetMangaModel
from src.models.resampler import Resampler
from src.models.qwen_resampler import QwenResampler
from src.models.mllm.seed_x import ContinuousLVLM
from src.models.mllm.modeling_llama_xformer import LlamaForCausalLM
from src.pipelines.pipeline_diffsensei import DiffSenseiPipeline


def result_generation(
    pipeline,
    prompt,
    height,
    width,
    num_samples,
    seed,
    ip_images,
    ip_bbox,
    dialog_bbox,
    num_inference_steps,
    guidance_scale,
    negative_prompt,
    ip_scale,
):
    generator = torch.Generator('cuda:0').manual_seed(seed)

    images = pipeline(
        prompt=prompt,
        prompt_2=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        num_samples=num_samples,
        generator=generator,
        # manga conditions
        ip_images=ip_images,
        ip_bbox=ip_bbox.copy(),
        ip_scale=ip_scale,
        dialog_bbox=dialog_bbox.copy(),
    ).images

    return images

def process_bounding_boxes(click_img, canvas_width, canvas_height, original_width, original_height):
    img, points = None, np.array([[[0, 0, 4]]])
    if click_img is not None:
        img, points = click_img["image"], click_img["points"]
        points = np.array(points).reshape((-1, 2, 3))
        points = points.reshape((-1, 3))
        lt = points[np.where(points[:, 2] == 2)[0]][None, :, :]
        rb = points[np.where(points[:, 2] == 3)[0]][None, :, :]
        poly = points[np.where(points[:, 2] <= 1)[0]][None, :, :]
        if len(lt) > 0 and len(rb) > 0:
            points = [lt, rb, poly] if len(lt) > 0 else [poly, np.array([[[0, 0, 4]]])]
            points = np.concatenate(points, axis=1)
        else:
            points = np.array([[[0, 0, 4]]])  # Default points if lt or rb are empty
    
    points = np.array([[[0, 0, 4]]]) if (len(points) == 0 or points.size == 0) else points
    bboxes = []
    for i in range(len(lt[0])):
        bbox = [lt[0][i], rb[0][i]]
        bboxes.append(bbox)

    for bbox in bboxes:
        print(f"bbox: {bbox}")

    # Convert bounding boxes to the desired format and scale them
    processed_bboxes = []
    for bbox in bboxes:
        lt, rb = bbox
        x1, y1 = float(lt[0]) * original_width / canvas_width, float(lt[1]) * original_height / canvas_height
        x2, y2 = float(rb[0]) * original_width / canvas_width, float(rb[1]) * original_height / canvas_height
        processed_bboxes.append([x1, y1, x2, y2])

    for bbox in processed_bboxes:
        print(f"processed_bbox: {bbox}")

    return processed_bboxes


def load_images(file_paths):
    return [Image.open(file_path).convert("L").convert("RGB") for file_path in file_paths]

def create_blank_image_dict(width=1024, height=1024):
    image = Image.new('RGB', (width, height), color='white')
    return {"image": image, "points": []}

def create_dialog_image_dict(ip_bbox_image, original_width, original_height, canvas_width=1024, canvas_height=1024):
    image = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(image)
    for bbox in ip_bbox_image.get('points', []):
        print("bbox", bbox)
        assert len(bbox) == 4, "Expected bbox to have 4 elements (x1, y1, x2, y2)"
        x1, y1, x2, y2 = bbox
        # Scale bbox coordinates to canvas size
        x1, y1 = x1 * canvas_width / original_width, y1 * canvas_height / original_height
        x2, y2 = x2 * canvas_width / original_width, y2 * canvas_height / original_height
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
    return {"image": image, "points": []}


def calculate_canvas_height(slider_width, slider_height, fixed_width=1024):
    return int((slider_height / slider_width) * fixed_width)


def main(args):
    # Load config
    config = OmegaConf.load(args.config_path)
    inference_config = OmegaConf.load(args.inference_config_path)

    # Load models
    weight_dtype = torch.float16
    unet = UNetMangaModel.from_pretrained(os.path.join(args.ckpt_path, "image_generator"), subfolder="unet", torch_dtype=weight_dtype)
    unet.set_manga_modules(
        max_num_ips=config.image_generator.max_num_ips,
        num_vision_tokens=config.image_generator.num_vision_tokens,
        max_num_dialogs=config.image_generator.max_num_dialogs,
    )
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(os.path.join(args.ckpt_path, "image_generator", "clip_image_encoder"), torch_dtype=weight_dtype)
    magi_image_encoder = ViTMAEModel.from_pretrained(os.path.join(args.ckpt_path, "image_generator", "magi_image_encoder"), torch_dtype=weight_dtype).to(device="cuda:0")
    
    image_proj_model = Resampler(
        dim=config.resampler.dim,
        depth=config.resampler.depth,
        dim_head=config.resampler.dim_head,
        heads=config.resampler.heads,
        num_queries=config.image_generator.num_vision_tokens,
        num_dummy_tokens=config.image_generator.num_dummy_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=config.resampler.ff_mult,
        magi_embedding_dim=magi_image_encoder.config.hidden_size
    ).to(device='cuda:0', dtype=weight_dtype)
    checkpoint = torch.load(os.path.join(args.ckpt_path, "image_generator", "image_proj_model", "pytorch_model.bin"), map_location='cpu')
    image_proj_model.load_state_dict(checkpoint)

    # Load MLLM
    tokenizer_mllm = LlamaTokenizer.from_pretrained(os.path.join(args.ckpt_path, "mllm", "tokenizer"))
    llm_model = LlamaForCausalLM.from_pretrained(os.path.join(args.ckpt_path, "mllm", "llm"), torch_dtype=weight_dtype)

    input_resampler = QwenResampler(**config.agent.input_resampler)
    output_resampler = QwenResampler(**config.agent.output_resampler)
    agent_model = ContinuousLVLM.from_pretrained(
        llm=llm_model,
        input_resampler=input_resampler,
        output_resampler=output_resampler,
    ).to(device='cuda:0', dtype=weight_dtype)
    
    # Load checkpoint weights
    checkpoint = torch.load(os.path.join(args.ckpt_path, "mllm", "agent", "pytorch_model.bin"))
    agent_model.load_state_dict(checkpoint, strict=False)

    pipeline = DiffSenseiPipeline.from_pretrained(
        os.path.join(args.ckpt_path, "image_generator"),
        unet=unet,
        image_encoder=image_encoder,
        torch_dtype=weight_dtype,
    )
    pipeline.progress_bar_config = {"disable": True}
    pipeline.register_manga_modules(
        image_proj_model=image_proj_model,
        magi_image_encoder=magi_image_encoder
    )
    pipeline.to(device='cuda:0', dtype=weight_dtype)

    print(f"All models and pipelines load complete")

    exit()

    import gradio_image_prompter as gr_ext

    # Define Gradio inputs and outputs
    inputs = [
        gr.Textbox(label="Prompt", lines=1, value="Enter prompt here"),
        gr.Slider(label="Height", minimum=128, maximum=2048, step=8, value=512),
        gr.Slider(label="Width", minimum=128, maximum=2048, step=8, value=512),
        gr.Slider(label="Num Samples", minimum=1, maximum=8, step=1, value=inference_config.num_samples),
        gr.Number(label="Random Seed", value=0, precision=0, minimum=0, maximum=99999999),
        gr.File(label="IP Images", file_count="multiple", type="filepath"),
        gr_ext.ImagePrompter(label="IP BBox"),
        gr_ext.ImagePrompter(label="Dialog BBox"),
        gr.Slider(label="Num Inference Steps", minimum=1, maximum=100, step=1, value=inference_config.num_inference_steps),
        gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, step=0.5, value=inference_config.guidance_scale),
        gr.Textbox(label="Negative Prompt", lines=1, value=inference_config.negative_prompt),
        gr.Slider(label="IP Scale", minimum=0.0, maximum=1.0, step=0.1, value=inference_config.ip_scale)
    ]
    outputs = gr.Gallery(label="Generated Images")

    # Custom function to generate the blank image dict
    def generate_blank_image_dict_and_run(*args):
        (
            prompt, height, width, num_samples, seed, ip_images,
            ip_bbox, dialog_bbox, num_inference_steps, guidance_scale,
            negative_prompt, ip_scale
        ) = args

        # Calculate canvas dimensions according to the fixed width of 1024
        canvas_width = 1024
        canvas_height = calculate_canvas_height(width, height, canvas_width)

        return result_generation(
            pipeline=pipeline,
            prompt=prompt,
            height=height,
            width=width,
            num_samples=num_samples,
            seed=seed,
            ip_images=load_images(ip_images) if ip_images else [],
            ip_bbox=process_bounding_boxes(ip_bbox, canvas_width, canvas_height, width, height) if ip_bbox else [],
            dialog_bbox=process_bounding_boxes(dialog_bbox, canvas_width, canvas_height, width, height) if dialog_bbox else [],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            ip_scale=ip_scale,
        )

    # Function to create blank image dict based on slider values
    def create_dynamic_blank_image(height, width):
        canvas_width = 1024
        canvas_height = calculate_canvas_height(width, height, canvas_width)
        return create_blank_image_dict(canvas_width, canvas_height)

    # Function to update ImagePrompter based on slider values
    def update_image_prompter_dims(height, width):
        canvas_width = 1024
        canvas_height = calculate_canvas_height(width, height, canvas_width)
        return gr_ext.ImagePrompter(label="IP BBox", value=create_blank_image_dict(canvas_width, canvas_height), width=canvas_width, height=canvas_height), \
               gr_ext.ImagePrompter(label="Dialog BBox", value=create_blank_image_dict(canvas_width, canvas_height), width=canvas_width, height=canvas_height)

    # Function to update Dialog ImagePrompter based on IP BBox
    def update_dialog_bbox(ip_bbox, height, width):
        canvas_width = 1024
        canvas_height = calculate_canvas_height(width, height, canvas_width)
        processed_ip_bbox = process_bounding_boxes(ip_bbox, canvas_width, canvas_height, width, height)
        return create_dialog_image_dict({"image": None, "points": processed_ip_bbox}, width, height, canvas_width, canvas_height)

    # Create Gradio interface
    with gr.Blocks(title="Manga Generation Demo") as demo:
        prompt = gr.Textbox(label="Prompt", lines=1, value="Enter prompt here")
        height = gr.Slider(label="Height", minimum=128, maximum=2048, step=8, value=512)
        width = gr.Slider(label="Width", minimum=128, maximum=2048, step=8, value=512)
        num_samples = gr.Slider(label="Num Samples", minimum=1, maximum=8, step=1, value=inference_config.num_samples)
        seed = gr.Number(label="Random Seed", value=0, precision=0, minimum=0, maximum=99999999)
        ip_images = gr.File(label="IP Images", file_count="multiple", type="filepath")
        ip_bbox = gr_ext.ImagePrompter(label="IP BBox", value=create_blank_image_dict(), width=1024, height=calculate_canvas_height(512, 512, 1024))
        end_ip_bbox_button = gr.Button("End Drawing IP BBox")
        dialog_bbox = gr_ext.ImagePrompter(label="Dialog BBox", value=create_blank_image_dict(), width=1024, height=calculate_canvas_height(512, 512, 1024))
        num_inference_steps = gr.Slider(label="Num Inference Steps", minimum=1, maximum=100, step=1, value=inference_config.num_inference_steps)
        guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, step=0.5, value=inference_config.guidance_scale)
        negative_prompt = gr.Textbox(label="Negative Prompt", lines=1, value=inference_config.negative_prompt)
        ip_scale = gr.Slider(label="IP Scale", minimum=0.0, maximum=1.0, step=0.1, value=inference_config.ip_scale)

        # Update blank image size dynamically
        height.change(fn=update_image_prompter_dims, inputs=[height, width], outputs=[ip_bbox, dialog_bbox])
        width.change(fn=update_image_prompter_dims, inputs=[height, width], outputs=[ip_bbox, dialog_bbox])

        end_ip_bbox_button.click(
            fn=update_dialog_bbox,
            inputs=[ip_bbox, height, width],
            outputs=[dialog_bbox]
        )

        generate_button = gr.Button("Generate Images")
        generated_images = gr.Gallery(label="Generated Images")

        generate_button.click(
            fn=generate_blank_image_dict_and_run,
            inputs=[
                prompt, height, width, num_samples, seed, ip_images,
                ip_bbox, dialog_bbox, num_inference_steps, guidance_scale,
                negative_prompt, ip_scale
            ],
            outputs=generated_images,
        )

    demo.launch()

if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=7 \
    python -m scripts.demo.gradio \
        --config_path configs/model/diffsensei.yaml \
        --inference_config_path configs/inference/diffsensei.yaml \
        --ckpt_path checkpoints/diffsensei
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--inference_config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
