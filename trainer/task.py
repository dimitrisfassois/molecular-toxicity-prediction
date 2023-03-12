"""Executes model training and evaluation."""
import argparse
import logging
import os
import sys

import deepchem as dc

from util.constants import CONST
from data_loaders.data_loaders_utils import get_generator
from .model_training import get_model
from .model_evaluation import evaluate_model


def run_experiment(flags):
    """Testbed for running model training and evaluation."""
    logging.info("Loading data for training and evaluation.")
    train_dataset = dc.data.DiskDataset(flags.train_data_dir)
    val_dataset = dc.data.DiskDataset(flags.val_data_dir)

    logging.info(f"Initializing model: {flags.model_type}")
    n_tasks = len(CONST.TASKS)
    dc_model = get_model(flags.model_type, n_tasks, flags.batch_size, flags.model_dir)

    logging.info(f"Starting model training for {flags.n_epochs} epochs.")
    train_generator = get_generator(
        flags.model_type, train_dataset, flags.batch_size, flags.n_epochs, n_tasks
    )
    avg_loss = dc_model.fit_generator(train_generator)
    logging.info(f"Average loss over the most recent checkpoint interval: {avg_loss}")

    logging.info("Evaluating model on eval dataset.")
    val_generator = get_generator(
        flags.model_type,
        val_dataset,
        flags.batch_size,
        1,
        n_tasks,
        deterministic=True,
    )
    evaluate_model(dc_model, val_generator, val_dataset)


def _parse_args():
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()

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
        "--model_dir",
        help="Output directory for saving model checkpoints.",
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

    return parser.parse_args()


def main():
    """Entry point."""

    flags = _parse_args()
    logging.basicConfig(level=flags.log_level.upper())
    run_experiment(flags)


if __name__ == "__main__":
    main()
