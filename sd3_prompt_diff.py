from diffusers.models import attention_processor
from attn_processor import xFuserJointAttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.JointAttnProcessor2_0 = CustomAttnProcessor2_0

import matplotlib.pyplot as plt
from diffusers import StableDiffusion3Pipeline
import os
import torch
import torch.distributed

def main():
    # SD3 model has 24 == 4x6 transformer blocks
    row, column = 4, 6
    relative = False
    os.makedirs('figs/prompt_diff', exist_ok=True)
    os.makedirs('results/prompt_diff', exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path="/cfs/dit/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()

    num_inference_steps = 20
    output1 = pipe(
        prompt="a photo of a cat holding a sign that says hello world",
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(123),
    ).images[0]
    
    output2 = pipe(
        prompt="chinese stock on 4000 points",
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(123),
    ).images[0]


    fig1, ax1 = plt.subplots(row, column, figsize=(32, 16))
    fig2, ax2 = plt.subplots(row, column, figsize=(32, 16))
        
    transformer = pipe.transformer
    for i, layer in enumerate(transformer.transformer_blocks):
        if hasattr(layer, 'attn'):
            print(f"ploting layer {i}: {type(layer).__name__}")
            layer.attn.processor.plot_kv_diff_prompts(i, ax1, column, relative=relative)
            layer.attn.processor.plot_activation_diff_prompts(i, ax2, column, relative=relative)

    fig1.tight_layout()
    fig2.tight_layout()

    relative_name = 'relative' if relative else 'absolute'
    fig1.savefig(os.path.join('figs/prompt_diff', f'sd3_kv_diffs_20_steps_{relative_name}.png'))
    fig2.savefig(os.path.join('figs/prompt_diff', f'sd3_activation_diffs_20_steps_{relative_name}.png'))
    output1.save(os.path.join('results/prompt_diff', f'sd3_output_20_steps_1_{relative_name}.png'))
    output2.save(os.path.join('results/prompt_diff', f'sd3_output_20_steps_2_{relative_name}.png'))


if __name__ == "__main__":
    main()
