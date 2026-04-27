import argparse
from pathlib import Path

import numpy as np
from pytorch_lightning import Trainer, seed_everything

from beat_this.dataset import BeatDataModule
from beat_this.dataset.dataset import BeatTrackingDataset
from beat_this.inference import load_checkpoint
from beat_this.model.pl_module import PLBeatThis

import wandb
import inspect

# for repeatability
seed_everything(0, workers=True)


def main(args):
    print("Single model prediction for", args.model)
    checkpoint = load_checkpoint(args.model)

    # create datamodule
    datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)
    # create model and trainer
    model, trainer = plmodel_setup(
        checkpoint,
        args.eval_trim_beats,
        args.dbn,
        args.gpu,
        segment_metrics=args.segment_metrics,
        causal_inference=args.causal_inference,
        sclm_lookahead_size=args.sclm_lookahead_size,
    )
    # predict
    metrics, dataset, preds, piece = compute_predictions(model, trainer, datamodule.predict_dataloader())

    # compute averaged metrics
    averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}

    # compute metrics averaged by dataset
    dataset_metrics = {
        k: {d: np.mean(v[dataset == d]) for d in np.unique(dataset)}
        for k, v in metrics.items()
    }

    # print for dataset
    print("Metrics")
    for k, v in averaged_metrics.items():
        print(f"{k}: {v}")
    print("Dataset metrics")
    for k, v in dataset_metrics.items():
        print(k)
        for d, value in v.items():
            print(f"{d}: {value}")
        print("------")

    if args.update_wandb:
        print(f'Attempting to update wandb metrics for run: {args.update_wandb}')
        try:
            api = wandb.Api()
            run_path = 'bank_juhasz_msc_thesis/beat_this/' + args.update_wandb
            run = api.run(run_path)

            if args.segment_metrics:
                to_be_updated = averaged_metrics
            else:
                to_be_updated = {
                    "F-measure_beat",
                    "F-measure_downbeat",
                    "Precision_beat",
                    "Precision_downbeat",
                    "Recall_beat",
                    "Recall_downbeat",
                }

            filtered_metrics = {k: v for k, v in averaged_metrics.items() if k in to_be_updated}
            upload_package = {f"{args.datasplit}_{k}": v for k, v in filtered_metrics.items()}  # adding datasplit for clarity

            if not upload_package:
                print("No metrics found to upload.")
            else:
                run.summary.update(upload_package)
                print(f"Uploaded metrics: {sorted(upload_package.keys())}")

        except Exception as e:
            print(f'wandb metrics update failed due to the following error: {e}')


def datamodule_setup(checkpoint, num_workers, datasplit):
    print("Creating datamodule")
    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    datamodule_hparams = checkpoint["datamodule_hyper_parameters"]
    # update the hparams with the ones from the arguments
    if num_workers is not None:
        datamodule_hparams["num_workers"] = num_workers
    datamodule_hparams["predict_datasplit"] = datasplit
    datamodule_hparams["data_dir"] = data_dir

    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage="predict")

    return datamodule


def plmodel_setup(
        checkpoint,
        eval_trim_beats,
        dbn,
        gpu,
        segment_metrics=False,
        causal_inference=False,
        sclm_lookahead_size=None,
):
    """
    Set up the pytorch lightning model and trainer for evaluation.

    Args:
        checkpoint (dict): The dict containing the checkpoint to load.
        eval_trim_beats (int or None): The number of beats to trim during evaluation. If None, the setting is taken from the pretrained model.
        dbn (bool or None): Whether to use the Dynamic Bayesian Network (DBN) module during evaluation. If None, the default behavior from the pretrained model is used.
        gpu (int): The index of the GPU device to use for predictions.
        segment_metrics (bool): Whether to compute metrics in 10s segments per excerpt.
        causal_inference (bool): Whether to compute metrics without stitching and to use causal postp.
        sclm_window_size (int): Size of lookahead in SCLM postprocessing for label-shifted models.
    Returns:
        tuple: A tuple containing the initialized pytorch lightning model and trainer.
    """
    if eval_trim_beats is not None:
        checkpoint["hyper_parameters"]["eval_trim_beats"] = eval_trim_beats
    if dbn is not None:
        checkpoint["hyper_parameters"]["use_dbn"] = dbn
    checkpoint["hyper_parameters"]["segment_metrics"] = segment_metrics
    checkpoint["hyper_parameters"]["causal_inference"] = causal_inference
    checkpoint["hyper_parameters"]["sclm_lookahead_size"] = sclm_lookahead_size

    # avoiding errors due to checkpoints containing arguments for the constructor of PLBeatThis which are not there
    # anymore. we only import those which are actually expected.
    sig = inspect.signature(PLBeatThis.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered_hparams = {k: v for k, v in checkpoint["hyper_parameters"].items() if k in allowed}
    dropped = sorted(set(checkpoint["hyper_parameters"]) - set(filtered_hparams))
    if dropped:
        print("Dropped hparams not in current PLBeatThis.__init__:", dropped)

    model = PLBeatThis(**filtered_hparams)
    model.load_state_dict(checkpoint["state_dict"])

    # trying to compile
    #model.model = torch.compile(model.model)

    # set correct device and accelerator
    if gpu >= 0:
        devices = [gpu]
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"
    # create trainer
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=None,
        deterministic=True,
        precision="16-mixed",
    )
    return model, trainer


def compute_predictions(model, trainer, predict_dataloader):
    print("Computing predictions ...")
    out = trainer.predict(model, predict_dataloader)

    # during segmented evaluation, it can happen that no metrics are calculated due to the excerpt being too short
    # in these cases, empty dicts are returned for metrics. we therefore exclude these pieces entirely and remove them
    # from the returned lists.
    filtered = [
        (m, p, d, pc)
        for (m, p, d, pc) in out
        if m  # keeps only truthy m, i.e. non‐empty dict
    ]

    if filtered:
        metrics, preds, dataset, piece = zip(*filtered)
        metrics = list(metrics)
        preds = list(preds)
        dataset = np.asarray([dlist[0] for dlist in dataset])
        piece = np.asarray([pclist[0] for pclist in piece])
    else:
        raise Exception("No valid metrics found.")

    # convert metrics from list of per-batch dictionaries to a single dictionary with np arrays as values
    metrics = {k: np.asarray([m[k] for m in metrics]) for k in metrics[0]}

    return metrics, dataset, preds, piece


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes predictions for a given model and dataset, "
        "prints metrics, and optionally dumps predictions to a given file."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Local checkpoint file. Only single model evaluations are supported.",
    )
    parser.add_argument(
        "--datasplit",
        type=str,
        choices=("train", "val", "test"),
        default="val",
        help="data split to use: train, val or test " "(default: %(default)s)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--num_workers", type=int, default=8, help="number of data loading workers "
    )
    parser.add_argument(
        "--eval_trim_beats",
        metavar="SECONDS",
        type=float,
        default=None,
        help="Override whether to skip the first given seconds "
        "per piece in evaluating (default: as stored in model)",
    )

    parser.add_argument(
        "--aggregation-type",
        type=str,
        choices=("mean-std", "k-fold"),
        default="mean-std",
        help="Type of aggregation to use for multiple models; ignored if only one model is given",
    )
    parser.add_argument(
        "--segment_metrics",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If True, compute metrics separately for 3 10s segments within each 30s excerpt."
    )
    parser.add_argument(
        "--update_wandb",
        type=str,
        default=None,
        help="W&B run ID of the run which the evaluation metrics will be uploaded to."
    )
    parser.add_argument(
        "--dbn",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="override the option to use madmom postprocessing dbn",
    )
    parser.add_argument(
        "--causal_inference",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Perform the appropriate causal postprocessing for the model variant."
    )
    parser.add_argument(
        "--sclm_lookahead_size",
        default=None,
        type=int,
        help="Lookahead size for SCLM postprocessing (default: shift size of trained model)."
             "By choosing a lookahead size < shift size, we can predict with 0 or negative latency.)",
    )

    args = parser.parse_args()

    main(args)
