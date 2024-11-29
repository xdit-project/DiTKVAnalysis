# DiT Cache Analysis

The repo is for analyzing the difference in K/V/activations for DiT models across diffusion steps.


## Installation

```shell
pip install -r requirements.txt

# For Open-Sora
git clone https://github.com/hpcaitech/Open-Sora.git
cd Open-Sora/
git reset --hard 476b6dc
git apply ../OpenSoraCacheAnalysis.patch
pip install -v -e .
cd ..

# For Mochi
git clone https://github.com/xdit-project/mochi-xDiT.git
cd mochi-xDiT/
git reset --hard d15495d
git apply ../MochiCacheAnalysis.patch
pip install -e .
cd ..
```

## Run Experiments

```shell
bash ./run_experiments.sh
```

## Draw Figures

```shell
mkdir exp-results/
python exp_1_varying_prompt.py
python exp_2_varying_model.py
```

## Citations

[Unveiling Redundancy in Diffusion Transformers (DiTs): A Systematic Study](https://arxiv.org/abs/2411.13588)

```
@article{sun2024unveiling,
  title={Unveiling Redundancy in Diffusion Transformers (DiTs): A Systematic Study},
  author={Sun, Xibo and Fang, Jiarui and Li, Aoyu and Pan, Jinzhe},
  journal={arXiv preprint arXiv:2411.13588},
  year={2024}
}
```
