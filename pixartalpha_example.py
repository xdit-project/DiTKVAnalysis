from diffusers.models import attention_processor
from attn_processor import AttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.AttnProcessor2_0 = CustomAttnProcessor2_0

from diffusers import PixArtAlphaPipeline
import torch
import numpy as np

def main():
    path = 'redundancy/pixart-alpha'
    pipe = PixArtAlphaPipeline.from_pretrained(
        pretrained_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    with open('caption1000.txt') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line[:-1]
        output = pipe(
            prompt=line,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        transformer = pipe.transformer
        for layer_index, layer in enumerate(transformer.transformer_blocks):
            processor = layer.attn1.processor
            np.save(f'{path}/{i}_l{layer_index}', processor.info)
            processor.reset_cache()
        output.save(f'results/pixart-alpha/{i}.jpg')
        print("finish", i)

if __name__ == "__main__":
    main()
