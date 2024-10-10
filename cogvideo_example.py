from diffusers.models import attention_processor
from diffusers.utils import export_to_video
from attn_processor import xFuserCogVideoXAttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.CogVideoXAttnProcessor2_0 = CustomAttnProcessor2_0

# from diffusers import schedulers
# schedulers.CogVideoXDDIMScheduler = schedulers.DPMSolverMultistepScheduler

import matplotlib.pyplot as plt
from diffusers import CogVideoXPipeline
import os
import torch
import torch.distributed

def main():
    # CogVideoX model has 30 == 5x6 transformer blocks
    row, column = 5, 6

    pipe = CogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path="/cfs/dit/CogVideoX-2b",
        torch_dtype=torch.float16,
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()

    for num_inference_steps in [20, 40, 80, 160]:
        output = pipe(
            height=480,
            width=720,
            prompt="a small cat is playing wit a toy car.",
            num_inference_steps=num_inference_steps,
            num_frames=9,
            generator=torch.Generator(device="cuda").manual_seed(123),
        ).frames[0]

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
        os.makedirs('figs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        fig1.savefig(os.path.join('figs', f'cogvideo_kv_diffs_{num_inference_steps}_steps.png'))
        fig2.savefig(os.path.join('figs', f'cogvideo_activation_stats_{num_inference_steps}_steps.png'))
        export_to_video(output, os.path.join('results', f'cogvideo_output_{num_inference_steps}_steps.mp4'), fps=8)


if __name__ == "__main__":
    main()
