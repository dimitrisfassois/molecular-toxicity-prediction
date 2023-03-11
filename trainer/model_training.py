"Module with functions to use for model training."
import deepchem as dc

from models.ConcatenationModels import SelfAttentionModel, MultiHeadedAttentionModel


def get_loss():
    """

    :return:
    """
    loss = dc.models.losses.CategoricalCrossEntropy()
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    return loss, metric


def get_model(model_type, n_tasks, batch_size, model_dir):
    """

    :param model_type:
    :param n_tasks:
    :param batch_size:
    :param model_dir:
    :return:
    """
    assert model_type in (
        "multi_headed_attention",
        "self_attention",
        "multi_task_classifier",
    ), "Invalid input for model type!"

    loss, metric = get_loss()
    if model_type == "multi_headed_attention":
        n_features = 1792
        model = MultiHeadedAttentionModel(n_tasks, n_features)
        dc_model = dc.models.KerasModel(model=model, loss=loss, model_dir=model_dir)
    elif model_type == "self_attention":
        n_features = 1792
        model = SelfAttentionModel(n_tasks, n_features, batch_size)
        dc_model = dc.models.KerasModel(model=model, loss=loss, model_dir=model_dir)
    elif model_type == "multi_task_classifier":
        n_features = 1024
        dc_model = dc.models.MultitaskClassifier(
            n_tasks=n_tasks, n_features=n_features, layer_sizes=[1000], model_dir=model_dir
        )
    return dc_model
