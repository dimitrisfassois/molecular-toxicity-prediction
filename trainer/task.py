"""Executes model training and evaluation."""
import argparse
import logging
import fsspec

import deepchem as dc
import torch

from util.constants import CONST
from data_loaders.data_loaders_utils import get_generator
from data_loaders.data_loaders import get_disk_dataset
from models.callbacks import ValidationCallback
from .model_training import get_model
from .model_evaluation import evaluate_model


def run_experiment(flags):
    """Testbed for running model training and evaluation."""
    logging.info("Loading training, validation and test datasets.")
    project_id = "molecular-toxicity-prediction"
    fs = fsspec.filesystem('gs', project=project_id)
    train_dataset = get_disk_dataset(fs, flags.train_data_dir)
    val_dataset = get_disk_dataset(fs, flags.val_data_dir)
    test_dataset = get_disk_dataset(fs, flags.test_data_dir)

    logging.info(f"Initializing model: {flags.model_type}")
    n_tasks = len(CONST.TASKS)
    learning_rate = dc.models.optimizers.ExponentialDecay(flags.lr, 0.9, 1000)
    dc_model = get_model(flags.model_type, n_tasks, flags.batch_size, flags.model_dir, learning_rate)

    logging.info(f"Starting model training for {flags.n_epochs} epochs.")
    train_generator = get_generator(
        flags.model_type, train_dataset, flags.batch_size, flags.n_epochs, n_tasks
    )
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    logging.info(f"Setting up early stopping in the {flags.save_dir} directory")
    callback = ValidationCallback(
        val_dataset,
        100,
        metric,
        n_tasks,
        flags.model_type,
        flags.batch_size,
        save_dir=flags.save_dir,
    )

    avg_loss = dc_model.fit_generator(train_generator, callbacks=callback)
    logging.info(f"Average loss over the most recent checkpoint interval: {avg_loss}")

    logging.info("Loading model with best validation loss.")
    dc_model.restore(model_dir=flags.save_dir)

    logging.info(f"Checkpoint restored from:, {dc_model.get_checkpoints(flags.save_dir)}")
    
    if flags.model_type == "multi_task_classifier":
        data = {
            'model_state_dict': dc_model.model.state_dict(),
            'optimizer_state_dict': dc_model._pytorch_optimizer.state_dict(),
            'global_step': dc_model._global_step
        }
        with fs.open(flags.save_dir, mode='wb') as f:
            torch.save(data, f)

    logging.info("Evaluating model on test dataset.")
    test_generator = get_generator(
        flags.model_type,
        test_dataset,
        flags.batch_size,
        1,
        n_tasks,
        deterministic=True,
    )
    evaluate_model(dc_model, test_generator, test_dataset)


def _parse_args():
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    parser.add_argument(
        "--model_type",
        help="Type of model to use.",
        choices=[
            "multi_headed_attention",
            "self_attention",
            "multi_task_classifier",
        ],
        default="self_attention",
    )

    parser.add_argument(
        "--train_data_dir",
        help="""Location of training data.
            """,
        required=True,
    )

    parser.add_argument(
        "--val_data_dir",
        help="""Location of evaluation data.
            """,
        required=True,
    )

    parser.add_argument(
        "--test_data_dir",
        help="""Location of test data.
            """,
        required=True,
    )

    parser.add_argument(
        "--model_dir",
        help="Output directory for saving model checkpoints.",
        required=True,
    )

    parser.add_argument(
        "--save_dir",
        help="Directory for early stopping for saving model with best validation loss.",
        required=True,
    )

    parser.add_argument(
        "--log_level",
        help="Logging level.",
        choices=[
            "DEBUG",
            "ERROR",
            "FATAL",
            "INFO",
            "WARN",
        ],
        default="INFO",
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size for model training.",
        default=50,
        type=int,
    )

    parser.add_argument(
        "--n_epochs",
        help="Maximum number of epochs for which to train the model.",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        help="Starting point for learning rate to use in scheduler.",
        type=float,
        default=0.0002,
    )

    return parser.parse_args()


def main():
    """Entry point."""

    flags = _parse_args()
    logging.basicConfig(level=flags.log_level.upper())
    run_experiment(flags)


if __name__ == "__main__":
    main()
