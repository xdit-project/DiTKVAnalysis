from diffusers.models import attention_processor
from diffusers.utils import export_to_gif
from attn_processor import LatteAttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.AttnProcessor2_0 = CustomAttnProcessor2_0

from diffusers import LattePipeline
import torch
import numpy as np

def main():
    path = 'redundancy/latte'
    pipe = LattePipeline.from_pretrained(
        pretrained_model_name_or_path="/cfs/dit/Latte-1",
        torch_dtype=torch.float16,
    ).to("cuda")
    for layer in pipe.transformer.transformer_blocks + pipe.transformer.temporal_transformer_blocks:
        layer.attn1.processor.record = True
    
    with open('caption1000.txt') as f:
        lines = f.readlines()

    for i in range(1000):
        print("start", i)
        line = lines[i][:-1]
        output = pipe(
            prompt=line,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        transformer = pipe.transformer
        for layer_index, layer in enumerate(transformer.transformer_blocks + transformer.temporal_transformer_blocks):
            processor = layer.attn1.processor
            np.save(f'{path}/{i}_l{layer_index}', processor.info)
            processor.reset_cache()
        export_to_gif(output, f'results/latte/{i}.gif', fps=8)
        print("finish", i)


if __name__ == "__main__":
    main()
