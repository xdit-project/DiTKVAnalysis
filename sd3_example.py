from diffusers.models import attention_processor
from attn_processor import xFuserJointAttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.JointAttnProcessor2_0 = CustomAttnProcessor2_0

from diffusers import StableDiffusion3Pipeline
import torch
import numpy as np

def main():
    path = 'redundancy/sd3'
    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
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
            processor = layer.attn.processor
            np.save(f'{path}/{i}_l{layer_index}', processor.info)
            processor.reset_cache()
        output.save(f'results/sd3/{i}.jpg')
        print("finish", i)


if __name__ == "__main__":
    main()
