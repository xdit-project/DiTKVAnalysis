from diffusers.models import attention_processor
from attn_processor import FluxAttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.FluxAttnProcessor2_0 = CustomAttnProcessor2_0

from diffusers import FluxPipeline
import torch
import numpy as np

def main():
    path = 'redundancy/flux'
    pipe = FluxPipeline.from_pretrained(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",
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
        for layer_index, layer in enumerate(transformer.transformer_blocks + transformer.single_transformer_blocks):
            processor = layer.attn.processor
            np.save(f'{path}/{i}_l{layer_index}', processor.info)
            processor.reset_cache()
        output.save(f'results/flux/{i}.jpg')
        print("finish", i)

if __name__ == "__main__":
    main()
