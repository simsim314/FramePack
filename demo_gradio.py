from diffusers_helper.hf_login import login # Assuming this is used elsewhere or for setup

import os

# HF_HOME setup
try:
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
except NameError:
    print("⚠️ Warning: __file__ not defined, HF_HOME might not be set as intended if not in a script.")

import gradio as gr
import torch
import traceback
import einops
import numpy as np
import argparse

from PIL import Image
from PIL.PngImagePlugin import PngInfo # Crucial for adding metadata
import json

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import gpu, get_cuda_free_memory_gb
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Monkeypatch for Gradio routing
from gradio import route_utils
from starlette.requests import Request as StarletteRequest
_original_gradio_get_api_call_path = route_utils.get_api_call_path
def patched_gradio_get_api_call_path(request: StarletteRequest) -> str:
    api_path = request.url.path
    return api_path
route_utils.get_api_call_path = patched_gradio_get_api_call_path
print("✅ Monkeypatch applied to gradio.route_utils.get_api_call_path.")

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
# Command line arg for default state of text encoder offload checkbox
parser.add_argument(
    "--offload-text-encoders-default",
    action='store_true',
    help="Set the default for 'Offload Text Encoders to CPU' checkbox to True."
)
args = parser.parse_args()

print(f"⚙️ Parsed Arguments: {args}")
device = gpu
cpu_device = torch.device("cpu")

free_mem_gb = get_cuda_free_memory_gb(device)
print(f"ℹ️ Free VRAM {free_mem_gb:.2f} GB on {device}")

print(f"🔄 Loading all models to GPU ({device})...")
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).to(device)
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).to(device)
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).to(device)
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).to(device)
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).to(device)
print(f"✅ Models loaded.")

vae.eval(); text_encoder.eval(); text_encoder_2.eval(); image_encoder.eval(); transformer.eval()
print("🔄 Enabling VAE slicing and tiling.")
vae.enable_slicing(); vae.enable_tiling()
print("✅ VAE slicing and tiling enabled.")
transformer.high_quality_fp32_output_for_inference = True
print('ℹ️ transformer.high_quality_fp32_output_for_inference = True')
vae.requires_grad_(False); text_encoder.requires_grad_(False); text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False); transformer.requires_grad_(False)
print("ℹ️ Model gradients disabled.")

stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)
print(f"ℹ️ Outputs will be saved to: {os.path.abspath(outputs_folder)}")
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache, mp4_crf, fps_output, text_encoders_offload_flag, decode_nth_section):
    # ... (Initial variable setup, job_id, png_metadata_values, text encoding, image processing, VAE encode initial, CLIP vision - all as in your last verified version) ...
    # This part (lines 70-150 approx from your last verified script) remains unchanged.
    # For brevity, I'll pick up just before the main loop.
    global text_encoder, text_encoder_2 
    
    print(f"ℹ️ Worker: Params - length:{total_second_length}s, latent_win:{latent_window_size}, steps:{steps}, fps:{fps_output}, text_enc_offload:{text_encoders_offload_flag}, decode_nth:{decode_nth_section}")

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    print(f"ℹ️ Worker: Calculated total_latent_sections: {total_latent_sections}")

    job_id = generate_timestamp()
    saved_input_image_filename = f'{job_id}.png'
    saved_input_image_filepath = os.path.join(outputs_folder, saved_input_image_filename)
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    print(f"ℹ️ Worker: Job ID {job_id} started.")

    try:
        png_metadata_values = {
            "prompt": prompt, "negative_prompt": n_prompt, "seed": str(seed),
            "total_second_length_setting": str(total_second_length),
            "latent_window_size": str(latent_window_size), "steps_per_segment": str(steps),
            "cfg_scale": str(cfg), "distilled_cfg_scale": str(gs), "cfg_rescale": str(rs),
            "use_teacache": str(use_teacache), "mp4_crf_setting_for_video": str(mp4_crf),
            "output_fps_setting_for_video": str(fps_output),
            "text_encoders_offload_active": str(text_encoders_offload_flag),
            "decode_every_nth_section_setting": str(decode_nth_section),
            "job_id": job_id
        }
        
        print("🔄 Worker: Preparing for text encoding...")
        if text_encoder.device != device: text_encoder.to(device)
        if text_encoder_2.device != device: text_encoder_2.to(device)
        if text_encoders_offload_flag: torch.cuda.empty_cache()
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1.0: llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else: llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if text_encoders_offload_flag:
            text_encoder.to(cpu_device); text_encoder_2.to(cpu_device); torch.cuda.empty_cache()
        print("✅ Worker: Text encoding complete.")
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        llama_attention_mask = llama_attention_mask.to(device); llama_attention_mask_n = llama_attention_mask_n.to(device)
        
        print("🔄 Worker: Image processing...")
        H_orig, W_orig, C_orig = input_image.shape
        height, width = find_nearest_bucket(H_orig, W_orig, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        pil_input_image = Image.fromarray(input_image_np)
        png_metadata_values["generation_resolution_for_video"] = f"({height}, {width})"
        png_metadata_values["original_input_image_dimensions_for_video"] = f"({H_orig}, {W_orig})"
        pnginfo = PngInfo(); 
        for key, value in png_metadata_values.items(): pnginfo.add_text(key, str(value))
        pil_input_image.save(saved_input_image_filepath, "PNG", pnginfo=pnginfo)
        print(f"✅ Worker: Input image saved with metadata to {saved_input_image_filepath}")
        input_image_pt = (torch.from_numpy(input_image_np).float() / 127.5 - 1).permute(2, 0, 1)[None, :, None].to(device)
        print("✅ Worker: Image processing complete. VAE encoding...")
        start_latent = vae_encode(input_image_pt, vae)
        print("✅ Worker: VAE encoding complete. CLIP Vision encoding...")
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        print("✅ Worker: CLIP Vision encoding complete. Adjusting dtypes...")
        llama_vec = llama_vec.to(transformer.dtype); llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype); clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        print("✅ Worker: Dtypes adjusted. Starting sampling...")

        rnd = torch.Generator(device=device).manual_seed(seed)
        num_frames_for_sampler = latent_window_size * 4 - 3
        print(f"ℹ️ Worker: num_frames_for_sampler (arg for sample_hunyuan): {num_frames_for_sampler}")

        history_latents_raw = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=vae.dtype, device=device)
        history_pixels = None 
        accumulated_raw_latent_frames = 0

        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        print(f"ℹ️ Worker: Using latent_paddings sequence of length: {len(latent_paddings)}")

        nonlocal_history_pixels_ref = {'ref': history_pixels}

        for i, latent_padding in enumerate(latent_paddings):
            section_index_human = i + 1 # 1-based index for messages
            is_last_section = latent_padding == 0 # True if this is the final section in the planned sequence
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                print("⚠️ Worker: Generation ended by user request."); stream.output_queue.push(('end', None)); return

            print(f"🔄 Worker: Processing latent section {section_index_human}/{len(latent_paddings)} (padding_size={latent_padding_size}, is_last_section={is_last_section})")

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0).to(device)
            clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            clean_latents_pre = start_latent
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents_raw[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps)
            
            def callback(d): # Callback for sampler progress
                preview_latents = d['denoised'] 
                preview_pixels_gpu = vae_decode_fake(preview_latents.to(vae.dtype))
                preview_np = (preview_pixels_gpu * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview_np = einops.rearrange(preview_np, 'b c t h w -> (b h) (t w) c')
                if stream.input_queue.top() == 'end': raise KeyboardInterrupt('User ends the task.')
                current_step = d['i'] + 1; percentage = int(100.0 * current_step / steps)
                hint = f'Sampling section {section_index_human} ({current_step}/{steps})'
                current_display_frames = nonlocal_history_pixels_ref['ref'].shape[2] if nonlocal_history_pixels_ref['ref'] is not None else 0
                desc_text = f'Video (last saved): {current_display_frames} frames, {(current_display_frames / fps_output) :.2f}s. Previewing section {section_index_human}...'
                stream.output_queue.push(('progress', (preview_np, desc_text, make_progress_bar_html(percentage, hint))))
                return
            
            generated_latents_raw_segment = sample_hunyuan(
                transformer=transformer, sampler='unipc', width=width, height=height, frames=num_frames_for_sampler,
                real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                num_inference_steps=steps, generator=rnd, prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n, device=device, dtype=transformer.dtype,
                image_embeddings=image_encoder_last_hidden_state, latent_indices=latent_indices,
                clean_latents=clean_latents.to(transformer.dtype), clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x.to(transformer.dtype), clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x.to(transformer.dtype), clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            temp_generated_latents_for_history = generated_latents_raw_segment
            if is_last_section:
                temp_generated_latents_for_history = torch.cat([start_latent.to(generated_latents_raw_segment.dtype), generated_latents_raw_segment], dim=2)

            accumulated_raw_latent_frames += int(temp_generated_latents_for_history.shape[2])
            history_latents_raw = torch.cat([temp_generated_latents_for_history.to(history_latents_raw.dtype), history_latents_raw], dim=2)
            
            # --- MODIFIED Conditional VAE Decode and Save Logic ---
            should_decode_this_section = False
            if is_last_section: # Always decode and save on the very last section
                should_decode_this_section = True
                print(f"ℹ️ Worker: Last section ({section_index_human}), performing full VAE decode and save.")
            elif decode_nth_section == 1: # Decode and save every section
                should_decode_this_section = True
                print(f"ℹ️ Worker: Section {section_index_human} (decode_nth=1), performing full VAE decode and save.")
            elif decode_nth_section > 1 and (section_index_human % decode_nth_section == 1):
                # Decode and save on the 1st, (N+1)th, (2N+1)th, etc., sections.
                should_decode_this_section = True
                print(f"ℹ️ Worker: Section {section_index_human} is a decode interval (decode_nth={decode_nth_section}, index % N == 1), performing full VAE decode and save.")
            # If decode_nth_section == 0, only the is_last_section condition will trigger a decode.
            
            if should_decode_this_section:
                # Decode ALL accumulated raw latents up to this point
                latents_to_decode_for_output = history_latents_raw[:, :, :accumulated_raw_latent_frames, :, :]
                print(f"🔄 Worker: Decoding all {latents_to_decode_for_output.shape[2]} accumulated raw latents for output...")
                
                decoded_pixels_gpu = vae_decode(latents_to_decode_for_output, vae)
                nonlocal_history_pixels_ref['ref'] = decoded_pixels_gpu.cpu() # This replaces the entire history_pixels
                
                history_pixels = nonlocal_history_pixels_ref['ref'] 
                print(f"✅ Worker: Full decode complete. Accumulated pixel video shape: {history_pixels.shape}")
                torch.cuda.empty_cache()

                output_filename = os.path.join(outputs_folder, f'{job_id}_{history_pixels.shape[2]}.mp4')
                save_bcthw_as_mp4(history_pixels, output_filename, fps=fps_output, crf=mp4_crf)
                print(f"✅ Worker: Video saved: {output_filename} (Total pixel frames: {history_pixels.shape[2]}).")
                stream.output_queue.push(('file', output_filename))
            else:
                print(f"ℹ️ Worker: Skipping full VAE decode and save for section {section_index_human}.")

            if is_last_section: break # Exit loop after processing the last section defined by latent_paddings
        print(f"✅ Worker: Sampling complete for job_id: {job_id}.")
    except KeyboardInterrupt: print(f"⚠️ Worker: Job {job_id} interrupted by user.")
    except Exception as e: print(f"❌ Worker ERROR for job_id {job_id}: {type(e).__name__} - {e}"); traceback.print_exc()
    finally:
        print(f"ℹ️ Worker: Pushing 'end' signal for job_id {job_id}. Attempting final CUDA cache clear."); 
        if text_encoders_offload_flag:
            if text_encoder.device != cpu_device: text_encoder.to(cpu_device)
            if text_encoder_2.device != cpu_device: text_encoder_2.to(cpu_device)
        torch.cuda.empty_cache()
        stream.output_queue.push(('end', None))
    return

# MODIFIED SIGNATURE: Added decode_nth_section
def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache, mp4_crf, fps_output, text_encoders_offload_flag, decode_nth_section):
    global stream
    print(f"🔄 Process function called. Input image provided: {input_image is not None}")
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), None 
    stream = AsyncStream()
    print(f"ℹ️ Process: New AsyncStream created.")
    # MODIFIED CALL: Pass decode_nth_section
    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache, mp4_crf, fps_output, text_encoders_offload_flag, decode_nth_section)
    output_filename = None; last_successful_video_path = None
    while True:
        flag, data = stream.output_queue.next()
        if flag != 'progress': print(f"🔄 Process: Received '{flag}' from worker.")
        if flag == 'file':
            output_filename = data; last_successful_video_path = output_filename 
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), last_successful_video_path
        elif flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update()
        elif flag == 'end':
            final_preview_visibility = gr.update(visible=False) if output_filename else gr.update()
            final_desc = gr.update() if output_filename else "Process ended."
            final_html = '' if output_filename else gr.update()
            # If no file was pushed because decode was only at end, output_filename might be None until the very end.
            # The last_successful_video_path should hold the path if any save occurred.
            print(f"✅ Process: Worker finished. Final output filename via stream: {output_filename}. Last saved path: {last_successful_video_path}")
            final_video_to_show = output_filename if output_filename else last_successful_video_path
            yield final_video_to_show, final_preview_visibility, final_desc, final_html, gr.update(interactive=True), gr.update(interactive=False), last_successful_video_path
            break
        else: print(f"⚠️ Process: Unknown flag '{flag}' received from worker.")

def end_process(): print(f"🔄 end_process called."); stream.input_queue.push('end')
def refresh_video_display(last_video_path):
    print(f"🔄 Refreshing video display with: {last_video_path}")
    return last_video_path if last_video_path and os.path.exists(last_video_path) else None

quick_prompts = [['The girl dances gracefully, with clear movements, full of charm.'], ['A character doing some simple body movements.']]
css = make_progress_bar_css(); 

if __name__ == "__main__":
    block = gr.Blocks(css=css).queue(); print("ℹ️ Gradio Blocks initialized.")
    with block:
        gr.Markdown('# FramePack'); last_video_path_state = gr.State(value=None)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                prompt = gr.Textbox(label="Prompt", value=''); example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)
                with gr.Row(): start_button = gr.Button(value="Start Generation"); end_button = gr.Button(value="End Generation", interactive=False)
                with gr.Group():
                    text_encoders_offload_checkbox = gr.Checkbox(label="Optimize VRAM: Offload Text Encoders to CPU", value=args.offload_text_encoders_default, info="If checked, text encoders are moved to CPU when not actively encoding text.")
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                    # ADDED UI CONTROL for decode frequency
                    decode_nth_section_input = gr.Number(label="Decode & Save Frequency", value=1, minimum=0, precision=0, info="Decode/save video every Nth section. 0 = only at the very end. 1 = every section (slowest).")
                    
                    n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False); seed = gr.Number(label="Seed", value=31337, precision=0)
                    total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=15, step=0.1)
                    latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Diffusion steps per segment. Default 25 is recommended.', visible=True)
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                    rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                    mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1)
                    fps_output = gr.Slider(label="Output Video FPS", minimum=1, maximum=120, value=30, step=1)
            with gr.Column():
                preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
                refresh_button = gr.Button(value="🔄 Refresh Last Video")
                gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling...'); progress_desc = gr.Markdown('', elem_classes='no-generating-animation'); progress_bar = gr.HTML('', elem_classes='no-generating-animation')
        gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')
        
        # MODIFIED IPS: Added decode_nth_section_input
        ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, use_teacache, mp4_crf, fps_output, text_encoders_offload_checkbox, decode_nth_section_input]
        outputs_for_process = [result_video, preview_image, progress_desc, progress_bar, start_button, end_button, last_video_path_state]
        start_button.click(fn=process, inputs=ips, outputs=outputs_for_process); end_button.click(fn=end_process)
        refresh_button.click(fn=refresh_video_display, inputs=[last_video_path_state], outputs=[result_video])
    
    print("ℹ️ Gradio UI definition complete.")
    effective_port = args.port if args.port is not None else 7860
    print(f"🚀 Launching Gradio app on {args.server}:{effective_port}...")
    block.launch(server_name=args.server, server_port=effective_port, share=args.share, inbrowser=args.inbrowser)
