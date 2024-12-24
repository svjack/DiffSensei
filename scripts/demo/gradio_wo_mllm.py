import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import torch
import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor, ViTImageProcessor, CLIPVisionModelWithProjection, LlamaTokenizer, ViTMAEModel
import numpy as np
import gc

import gradio as gr
import gradio_image_prompter as gr_ext

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "tmp")

from src.models.unet import UNetMangaModel
from src.models.resampler import Resampler
from src.pipelines.pipeline_diffsensei import DiffSenseiPipeline
from .examples import example_inputs_wo_mllm

clip_image_processor = CLIPImageProcessor()
magi_image_processor = ViTImageProcessor()

default_bbox_canvas_size = 386


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
    ip_scale
):  
    generator = torch.Generator('cuda:0').manual_seed(seed)

    try:
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
            ip_image_embeds=None,
            ip_bbox=ip_bbox.copy(),
            ip_scale=ip_scale,
            dialog_bbox=dialog_bbox.copy(),
        ).images
    except Exception as e:
        print(f"generation failed! image shape: [{width}, {height}] num_sample: {num_samples}. Probably OOM.")
        gc.collect()
        torch.cuda.empty_cache()

    return images


def process_bounding_boxes(click_img, canvas_width, canvas_height):
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

    # Convert bounding boxes to the desired format and scale them
    processed_bboxes = []
    for bbox in bboxes:
        # print(f"bbox: {bbox}")
        lt, rb = bbox
        x1, y1 = float(lt[0]) / canvas_width, float(lt[1]) / canvas_height
        x2, y2 = float(rb[0]) / canvas_width, float(rb[1]) / canvas_height
        processed_bboxes.append([x1, y1, x2, y2])

    for bbox in processed_bboxes:
        print(f"processed_bbox: {bbox}")

    return processed_bboxes


def load_images(file_paths):
    return [Image.open(file_path).convert("L").convert("RGB") for file_path in file_paths]


def create_blank_image_dict(width, height):
    image = Image.new('RGB', (width, height), color='white')
    return {"image": image, "points": []}


def create_dialog_image_dict(ip_bbox_image, canvas_width, canvas_height):
    image = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(image)
    for bbox in ip_bbox_image.get('points', []):
        # print("bbox", bbox)
        assert len(bbox) == 4, "Expected bbox to have 4 elements (x1, y1, x2, y2)"
        x1, y1, x2, y2 = bbox
        # Scale bbox coordinates to canvas size
        x1, y1 = x1 * canvas_width, y1 * canvas_height
        x2, y2 = x2 * canvas_width, y2 * canvas_height
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
    return {"image": image, "points": []}


def calculate_canvas_size(slider_width, slider_height, longer_size=default_bbox_canvas_size):
    if slider_width <= slider_height:
        height = longer_size
        width = int((slider_width / slider_height) * longer_size)
    else:
        width = longer_size
        height = int((slider_height / slider_width) * longer_size)
    
    return height, width


# Function to update ImagePrompter based on slider values
def update_image_prompter_dims(height, width):
    canvas_height, canvas_width = calculate_canvas_size(width, height)
    return gr_ext.ImagePrompter(label="IP BBox", value=create_blank_image_dict(canvas_width, canvas_height), width=canvas_width, height=canvas_height), \
            gr_ext.ImagePrompter(label="Dialog BBox", value=create_blank_image_dict(canvas_width, canvas_height), width=canvas_width, height=canvas_height)


# Function to update Dialog ImagePrompter based on IP BBox
def update_dialog_bbox(ip_bbox, height, width):
    canvas_height, canvas_width = calculate_canvas_size(width, height)
    processed_ip_bbox = process_bounding_boxes(ip_bbox, canvas_width, canvas_height)
    return create_dialog_image_dict({"image": None, "points": processed_ip_bbox}, canvas_width, canvas_height)


def main(args):
    # Load config
    config = OmegaConf.load(args.config_path)
    inference_config = OmegaConf.load(args.inference_config_path)

    # Load models
    weight_dtype = torch.float16
    unet = UNetMangaModel.from_config(os.path.join(args.ckpt_path, "image_generator"), subfolder="unet", torch_dtype=weight_dtype)
    unet.set_manga_modules(
        max_num_ips=config.image_generator.max_num_ips,
        num_vision_tokens=config.image_generator.num_vision_tokens,
        max_num_dialogs=config.image_generator.max_num_dialogs,
    )
    checkpoint = torch.load(os.path.join(args.ckpt_path, "image_generator", "unet", "pytorch_model.bin"))
    unet.load_state_dict(checkpoint)
    
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

    pipeline = DiffSenseiPipeline.from_pretrained(
        os.path.join(args.ckpt_path, "image_generator"),
        unet=unet,
        image_encoder=image_encoder,
        torch_dtype=weight_dtype,
    )
    pipeline.progress_bar_config = {"disable": True}
    pipeline.register_manga_modules(
        image_proj_model=image_proj_model,
        magi_image_encoder=magi_image_encoder,
    )
    pipeline.to(device='cuda:0', dtype=weight_dtype)

    print(f"All models and pipelines load complete")

    # Custom function to generate the blank image dict
    def generate_blank_image_dict_and_run(*args):
        (
            prompt, height, width, num_samples, seed, ip_images,
            ip_bbox, dialog_bbox, num_inference_steps, guidance_scale,
            negative_prompt, ip_scale
        ) = args

        canvas_height, canvas_width = calculate_canvas_size(width, height)

        return result_generation(
            pipeline=pipeline,
            prompt=prompt,
            height=height,
            width=width,
            num_samples=num_samples,
            seed=seed,
            ip_images=load_images(ip_images) if ip_images else [],
            ip_bbox=process_bounding_boxes(ip_bbox, canvas_width, canvas_height) if ip_bbox else [],
            dialog_bbox=process_bounding_boxes(dialog_bbox, canvas_width, canvas_height) if dialog_bbox else [],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            ip_scale=ip_scale,
        )

    # Create Gradio interface
    with gr.Blocks(title="DiffSensei Demo") as demo:
        with gr.Row():
            gr.Markdown(\
"""## TIPS:
1. Upload manga character images, draw IP bounding boxes (should be the same number of uploaded images), click 'End Drawing IP Bbox', draw dialog bounding boxes, and click Generaet Images.
2. You can adjust the image height and width freely, just like creating manga panels in diverse aspect ratios. The bounding box drawing panels will change automatically.
3. Some example inputs are given at bottom. Click on them to have a start.
4. You can refer to our [MangaZero dataset](https://huggingface.co/datasets/jianzongwu/MangaZero) to get more resources.
5. Be patient. Try more prompts, characters, and random seeds, and download your favored manga panels! 🤗""")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=1, value="Enter your prompt here")
                height = gr.Slider(label="Height", minimum=128, maximum=2048, step=8, value=256)
                width = gr.Slider(label="Width", minimum=128, maximum=2048, step=8, value=256)
                num_samples = gr.Slider(label="Num Samples", minimum=1, maximum=8, step=4, value=inference_config.num_samples)
                seed = gr.Number(label="Random Seed", value=0, precision=0, minimum=0, maximum=99999999)
                ip_images = gr.File(label="IP Images", file_count="multiple", type="filepath")
                with gr.Row():
                    with gr.Column():
                        ip_bbox = gr_ext.ImagePrompter(label="IP BBox", value=create_blank_image_dict(default_bbox_canvas_size, default_bbox_canvas_size), width=default_bbox_canvas_size, height=default_bbox_canvas_size)
                        end_ip_bbox_button = gr.Button("End Drawing IP BBox")
                    with gr.Column():
                        dialog_bbox = gr_ext.ImagePrompter(label="Dialog BBox", value=create_blank_image_dict(default_bbox_canvas_size, default_bbox_canvas_size), width=default_bbox_canvas_size, height=default_bbox_canvas_size)
                num_inference_steps = gr.Slider(label="Num Inference Steps", minimum=1, maximum=60, step=1, value=inference_config.num_inference_steps)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, step=0.5, value=inference_config.guidance_scale)
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=1, value=inference_config.negative_prompt)
                ip_scale = gr.Slider(label="IP Scale", minimum=0.0, maximum=1.0, step=0.1, value=inference_config.ip_scale)
                
                generate_button = gr.Button("Generate Images")
            with gr.Column():
                generated_images = gr.Gallery(label="Generated Images")
        with gr.Row():
            gr.Examples(examples=example_inputs_wo_mllm, inputs=[prompt, height, width, num_samples, seed, ip_images])

        height.change(fn=update_image_prompter_dims, inputs=[height, width], outputs=[ip_bbox, dialog_bbox])
        width.change(fn=update_image_prompter_dims, inputs=[height, width], outputs=[ip_bbox, dialog_bbox])

        end_ip_bbox_button.click(
            fn=update_dialog_bbox,
            inputs=[ip_bbox, height, width],
            outputs=[dialog_bbox]
        )
        
        generate_button.click(
            fn=generate_blank_image_dict_and_run,
            inputs=[
                prompt, height, width, num_samples, seed, ip_images,
                ip_bbox, dialog_bbox, num_inference_steps, guidance_scale,
                negative_prompt, ip_scale
            ],
            outputs=generated_images,
        )

    demo.launch(share = True)


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.demo.gradio_wo_mllm \
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
