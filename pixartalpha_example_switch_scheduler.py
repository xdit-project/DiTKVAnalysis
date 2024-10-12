from diffusers.models import attention_processor
from attn_processor import AttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.AttnProcessor2_0 = CustomAttnProcessor2_0

from diffusers import schedulers
schedulers.DPMSolverMultistepScheduler = schedulers.CogVideoXDDIMScheduler

import matplotlib.pyplot as plt
from diffusers import PixArtAlphaPipeline
import os
import torch
import torch.distributed

def main():
    # PixArt model has 28 == 4x7 transformer blocks
    row, column = 4, 7
    relative = False
    os.makedirs('figs/switch_scheduler', exist_ok=True)
    os.makedirs('results/switch_scheduler', exist_ok=True)

    pipe = PixArtAlphaPipeline.from_pretrained(
        pretrained_model_name_or_path="/cfs/dit/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    
    for num_inference_steps in [20, 40, 80, 160]:
        output = pipe(
            height=1024,
            width=1024,
            prompt="chinses stock on 4000 points",
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(123),
        ).images[0]

        fig1, ax1 = plt.subplots(row, column, figsize=(32, 16))
        fig2, ax2 = plt.subplots(row, column, figsize=(32, 16))

        transformer = pipe.transformer
        for i, layer in enumerate(transformer.transformer_blocks):
            if hasattr(layer, 'attn1'):
                print(f"ploting layer {i}: {type(layer).__name__}")
                layer.attn1.processor.plot_kv_diff(i, ax1, column)
                layer.attn1.processor.plot_activation_diff(i, ax2, column)
        
        fig1.tight_layout()
        fig2.tight_layout()

        relative_name = 'relative' if relative else 'absolute'
        fig1.savefig(os.path.join('figs/switch_scheduler', f'pixart_kv_stats_{num_inference_steps}_steps_{relative_name}.png'))
        fig2.savefig(os.path.join('figs/switch_scheduler', f'pixart_activation_stats_{num_inference_steps}_steps_{relative_name}.png'))
        output.save(os.path.join('results/switch_scheduler', f'pixart_output_{num_inference_steps}_steps_{relative_name}.png'))

if __name__ == "__main__":
    main()
