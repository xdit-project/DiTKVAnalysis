from diffusers.models import attention_processor
from attn_processor import xFuserCogVideoXAttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.CogVideoXAttnProcessor2_0 = CustomAttnProcessor2_0

import matplotlib.pyplot as plt
from diffusers import CogVideoXPipeline
import time
import os
import torch
import torch.distributed

def main():
    pipe = CogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path="/cfs/dit/CogVideoX-2b",
        torch_dtype=torch.float16,
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=480,
        width=720,
        prompt="a small cat is playing wit a toy car.",
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(123),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated()

    # CogVideoX model has 30 == 5x6 transformer blocks
    fig, ax1 = plt.subplots(5, 6, figsize=(32, 16))
        
    transformer = pipe.transformer
    for i, layer in enumerate(transformer.transformer_blocks):
        print(f"ploting layer {i}: {type(layer).__name__}")
        if hasattr(layer, 'attn1'):
            layer.attn1.processor.plot_kv_diff(i, ax1)
    
    fig.tight_layout()
    fig.savefig(os.path.join('results', f'all_kv_stats.png'))

    for i, image in enumerate(output.images):
        img_file = (
            f"./results/cogvideo_result.png"
        )
        image.save(img_file)
        print(img_file)

if __name__ == "__main__":
    main()
