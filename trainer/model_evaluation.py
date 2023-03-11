"""Module for model evaluation functions."""
import logging

import deepchem as dc

from util.constants import CONST

def evaluate_model(dc_model, val_generator, val_dataset):
    y_true = val_dataset.y
    w = val_dataset.w
    n_tasks = len(CONST.TASKS)
    y_pred = dc_model.predict_on_generator(val_generator)
    y_pred = y_pred[:val_dataset.y.shape[0]]
    metric = dc.metrics.roc_auc_score
    for i in range(n_tasks):
        score = metric(dc.metrics.to_one_hot(y_true[:, i]), y_pred[:, i], sample_weight=w[:, i])
        logging.info(f"Task: {CONST.TASKS[i]}, score: {score}")
