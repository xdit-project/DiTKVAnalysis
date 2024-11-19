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