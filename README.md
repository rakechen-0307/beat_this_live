# Beat This! Live

This is an ongoing project to develop a live version of the original state-of-the-art beat and downbeat tracker Beat
This! from the ISMIR 2024 paper "[Beat This! Accurate Beat Tracking Without DBN Postprocessing](https://arxiv.org/abs/2407.21658)"
by Francesco Foscarin, Jan Schlüter and Gerhard Widmer.

The original Beat This! repository is available [here](https://github.com/CPJKU/beat_this/tree/main).

## ISMIR Submission

This is the preliminary version of the training, inference and evaluation code for Beat This! Live. To use it, install
the requirements as documented in the Beat This! repository shown above.

Training
---------
To re-run training for Beat This! Live, for a single seed:
...
For Beat This! Live Large:
    python3 launch_scripts/train.py --seed=0 --transformer-dim=512 --logger=wandb --causal-transformer --causal-convolution --sw-attention-window-size=512 --train-length=1536 --label-shift=-1
For Beat This! Live Zero:
    python3 launch_scripts/train.py --seed=0 --transformer-dim=128 --logger=wandb --causal-transformer --causal-convolution --sw-attention-window-size=256 --train-length=1536 --label-shift=-3
For Beat This! Live:
    python3 launch_scripts/train.py --seed=0 --transformer-dim=128 --logger=wandb --causal-transformer --causal-convolution --sw-attention-window-size=512 --train-length=1536 --label-shift=-1
...

Please note that the automatically calculated validation and test set metrics are incorrect. To recreate paper metrics,
use launch_scripts/compute_paper_metrics.py.

Evaluation
-------
To re-run evaluation for a particular checkpoint:
...
For Beat This! Live Large:
    python3 launch_scripts/compute_paper_metrics.py --model "checkpoints/beat_this_live_large.ckpt" --datasplit test --causal_inference
For Beat This! Live Zero:
    python3 launch_scripts/compute_paper_metrics.py --model "checkpoints/beat_this_live_zero.ckpt" --datasplit test --causal_inference --sclm_lookahead_size=1
For Beat This! Live:
    python3 launch_scripts/compute_paper_metrics.py --model "checkpoints/beat_this_live.ckpt" --datasplit test --causal_inference
...

Note that due to space constraints, for reviewing we only provide a single seed for each model instead of three seeds, so results will not exactly match those in the paper.
