"""Utility functions for data loaders."""
from .data_loaders import *


def get_generator(model_type, dataset, batch_size, n_epochs, n_tasks, deterministic=False):
    if model_type in ("self_attention", "multi_headed_attention"):
        generator = full_dataset_generator(dataset, batch_size, n_epochs, n_tasks, deterministic)
    elif model_type == "multi_task_classifier":
        generator = fingerprints_generator(dataset, batch_size, n_epochs, n_tasks, deterministic)
    return generator
