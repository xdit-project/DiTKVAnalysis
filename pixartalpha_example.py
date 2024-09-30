from diffusers.models import attention_processor
from attn_processor import AttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.AttnProcessor2_0 = CustomAttnProcessor2_0

from diffusers import PixArtAlphaPipeline
import time
import os
import torch
import torch.distributed

def main():
    pipe = PixArtAlphaPipeline.from_pretrained(
        pretrained_model_name_or_path="/cfs/dit/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=1024,
        width=1024,
        prompt="chinses stock on 4000 points",
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(123),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated()

    transformer = pipe.transformer
    for i, layer in enumerate(transformer.transformer_blocks):
        print(f"ploting layer {i}: {type(layer).__name__}")
        if hasattr(layer, 'attn1'):
            layer.attn1.processor.plot_kv_diff(i)
    
    for i, image in enumerate(output.images):
        img_file = (
            f"./results/pixart_alpha_result.png"
        )
        image.save(img_file)
        print(img_file)

if __name__ == "__main__":
    main()
