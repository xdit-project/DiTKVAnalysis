from diffusers.models import attention_processor
from attn_processor import FluxAttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.FluxAttnProcessor2_0 = CustomAttnProcessor2_0

import matplotlib.pyplot as plt
from diffusers import FluxPipeline
import os
import torch
import torch.distributed

def main():
    # Flux model has 57 == 6x10-3 transformer blocks
    row, column = 6, 10

    pipe = FluxPipeline.from_pretrained(
        pretrained_model_name_or_path="/cfs/dit/FLUX.1-schnell",
        torch_dtype=torch.float16,
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    
    for num_inference_steps in [20, 40, 80, 160]:
        output = pipe(
            height=1024,
            width=1024,
            prompt="chinese stock on 4000 points, no people in the figure",
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(123),
        ).images[0]

        fig1, ax1 = plt.subplots(row, column, figsize=(48, 32))
        fig2, ax2 = plt.subplots(row, column, figsize=(48, 32))

        transformer = pipe.transformer
        for i, layer in enumerate(transformer.transformer_blocks + transformer.single_transformer_blocks):
            if hasattr(layer, 'attn'):
                print(f"ploting layer {i}: {type(layer).__name__}")
                layer.attn.processor.plot_kv_diff(i, ax1, column)
                layer.attn.processor.plot_activation_diff(i, ax2, column)
        
        fig1.tight_layout()
        fig2.tight_layout()
        os.makedirs('figs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        fig1.savefig(os.path.join('figs', f'flux_kv_stats_{num_inference_steps}_steps.png'))
        fig2.savefig(os.path.join('figs', f'flux_activation_stats_{num_inference_steps}_steps.png'))
        output.save(os.path.join('results', f'flux_output_{num_inference_steps}_steps.png'))

if __name__ == "__main__":
    main()
