from diffusers.models import attention_processor
from diffusers.utils import export_to_video
from attn_processor import xFuserCogVideoXAttnProcessor2_0 as CustomAttnProcessor2_0
attention_processor.CogVideoXAttnProcessor2_0 = CustomAttnProcessor2_0

from diffusers import CogVideoXPipeline
import torch
import numpy as np

def main():
    path = 'redundancy/cogvideox-5b'
    pipe = CogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path="THUDM/CogVideoX-5b",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    with open('caption1000.txt') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line[:-1]
        output = pipe(
            prompt=line,
            generator=torch.Generator(device="cuda").manual_seed(42),
            num_frames=9,
        ).frames[0]

        transformer = pipe.transformer
        for layer_index, layer in enumerate(transformer.transformer_blocks):
            processor = layer.attn1.processor
            np.save(f'{path}/{i}_l{layer_index}', processor.info)
            processor.reset_cache()
        export_to_video(output, f'results/cogvideox-5b/{i}.mp4', fps=8)
        print("finish", i)

if __name__ == "__main__":
    main()
