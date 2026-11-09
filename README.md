# Beat This! Live

[Beat This! Live](https://ismir2026program.ismir.net/poster_187.html#paper) is an online beat and downbeat tracker published
in ISMIR 2026.

The model is based on the original (then) state-of-the-art offline beat tracker Beat This! from the ISMIR 2024 paper "[Beat This! Accurate Beat Tracking Without DBN Postprocessing](https://arxiv.org/abs/2407.21658)"
by Francesco Foscarin, Jan Schlüter and Gerhard Widmer.

The original Beat This! repository is available [here](https://github.com/CPJKU/beat_this/tree/main).

Training
---------
To re-run training for Beat This! Live, for a single seed:
```
For Beat This! Live Large:
    python3 launch_scripts/train.py --seed=0 --transformer-dim=512 --logger=wandb --causal-transformer --causal-convolution --sw-attention-window-size=512 --train-length=1536 --label-shift=-1
For Beat This! Live Zero:
    python3 launch_scripts/train.py --seed=0 --transformer-dim=128 --logger=wandb --causal-transformer --causal-convolution --sw-attention-window-size=256 --train-length=1536 --label-shift=-3
For Beat This! Live:
    python3 launch_scripts/train.py --seed=0 --transformer-dim=128 --logger=wandb --causal-transformer --causal-convolution --sw-attention-window-size=512 --train-length=1536 --label-shift=-1
```

Please note that the automatically calculated validation and test set metrics shown during training are incorrect.
To recreate paper metrics, use launch_scripts/compute_paper_metrics.py (see below).

Evaluation
-------
To re-run evaluation for a particular checkpoint:
```
For Beat This! Live Large:
    python3 launch_scripts/compute_paper_metrics.py --model "checkpoints/beat_this_live_large.ckpt" --datasplit test --causal_inference
For Beat This! Live Zero:
    python3 launch_scripts/compute_paper_metrics.py --model "checkpoints/beat_this_live_zero.ckpt" --datasplit test --causal_inference --sclm_lookahead_size=1
For Beat This! Live:
    python3 launch_scripts/compute_paper_metrics.py --model "checkpoints/beat_this_live.ckpt" --datasplit test --causal_inference
```

Trained model checkpoints can be downloaded from [here](https://cloud.cp.jku.at/index.php/s/xe57reQoRJzrw9N).

Note that the paper metrics are calculated as an average based on 3 separate model runs with 3 different seeds. We provide
a single model checkpoint for each version (instead of 3), thus results will not match those found in the paper exactly.

Please also note the use of `--sclm_lookahead_size=1` in the case of Beat This! Live Zero -- a model which has a label
shift of 3 frames. Producing predictions with this flag forces the model to only peek a single frame ahead of the current
one, thus enabling (theoretically) negative latency predictions.

General Notes
-------
Although the training regime and the models themselves will be left as is, minor code clean-ups and a more extensive
documentation are still needed and will be added later on.

License
-------
The code and the published model weights are released under the [MIT license](https://github.com/CPJKU/beat_this_live/blob/main/LICENSE).
Note that some of the training files are fully copyrighted or under limited Creative Commons licenses, and it is up to
the user to assess whether this may impact their use case.

Citation
-------
To be added.