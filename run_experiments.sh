mkdir -p ./redundancy/flux
python flux_example.py

mkdir -p ./redundancy/pixart-alpha
python pixartalpha_example.py

mkdir -p ./redundancy/sd3
python sd3_example.py

mkdir -p ./redundancy/cogvideox-5b
python cogvideo_example.py

mkdir -p ./redundancy/latte
python latte_example.py

mkdir -p ./redundancy/opensora
python ./Open-Sora/scripts/inference.py Open-Sora/configs/opensora-v1-2/inference/sample.py

mkdir -p ./redundancy/mochi
CUDA_VISIBLE_DEVICES=0 python ./mochi-xDiT/demos/cli.py --model_dir='genmo/mochi-1-preview'
