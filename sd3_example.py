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
    os.makedirs('figs/overall', exist_ok=True)
    os.makedirs('results/overall', exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path="/cfs/dit/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    
    for num_inference_steps in [20, 40, 80, 160]:
        output = pipe(
            prompt="a photo of a cat holding a sign that says hello world",
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(123),
        ).images[0]

        fig1, ax1 = plt.subplots(row, column, figsize=(32, 16))
        fig2, ax2 = plt.subplots(row, column, figsize=(32, 16))
            
        transformer = pipe.transformer
        for i, layer in enumerate(transformer.transformer_blocks):
            if hasattr(layer, 'attn'):
                print(f"ploting layer {i}: {type(layer).__name__}")
                layer.attn.processor.plot_kv_diff(i, ax1, column, relative=relative)
                layer.attn.processor.plot_activation_diff(i, ax2, column, relative=relative)
        
        fig1.tight_layout()
        fig2.tight_layout()

        relative_name = 'relative' if relative else 'absolute'
        fig1.savefig(os.path.join('figs/overall', f'sd3_kv_diffs_{num_inference_steps}_steps_{relative_name}.png'))
        fig2.savefig(os.path.join('figs/overall', f'sd3_activation_diffs_{num_inference_steps}_steps_{relative_name}.png'))
        output.save(os.path.join('results/overall', f'sd3_output_{num_inference_steps}_steps_{relative_name}.png'))


if __name__ == "__main__":
    main()
