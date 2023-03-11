"""Module for data preprocessing."""
import numpy as np
import deepchem as dc
from transformers import AutoTokenizer, TFRobertaModel


tox21_tasks, tox21_datasets, tox21_transformers = dc.molnet.load_tox21(
    featurizer="ECFP", reload=False
)
tox21_train_dataset, tox21_valid_dataset, tox21_test_dataset = tox21_datasets


tox21_tasks, graph_tox21_datasets, graph_tox21_transformers = dc.molnet.load_tox21(
    featurizer="GraphConv", reload=False
)
(
    graph_tox21_train_dataset,
    graph_tox21_valid_dataset,
    graph_tox21_test_dataset,
) = graph_tox21_datasets


tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
roberta = TFRobertaModel.from_pretrained(
    "seyonec/PubChem10M_SMILES_BPE_450k", from_pt=True
)
roberta.trainable = False


max_train_length = len(max(tox21_train_dataset.ids.tolist(), key=len))
max_val_length = len(max(tox21_valid_dataset.ids.tolist(), key=len))
max_test_length = len(max(tox21_test_dataset.ids.tolist(), key=len))
max_len = max(max_train_length, max_val_length, max_test_length)


train_tokens = tokenizer(
    tox21_train_dataset.ids.tolist(),
    return_tensors="tf",
    padding=True,
    truncation=True,
    max_length=max_len,
)
train_embeddings = roberta(train_tokens).pooler_output


val_tokens = tokenizer(
    tox21_valid_dataset.ids.tolist(),
    return_tensors="tf",
    padding=True,
    truncation=True,
    max_length=max_len,
)
val_embeddings = roberta(val_tokens).pooler_output


test_tokens = tokenizer(
    tox21_test_dataset.ids.tolist(),
    return_tensors="tf",
    padding=True,
    truncation=True,
    max_length=max_len,
)
test_embeddings = roberta(test_tokens).pooler_output

final_train_X = np.concatenate(
    (
        tox21_train_dataset.X,
        train_embeddings.numpy(),
        np.expand_dims(graph_tox21_train_dataset.X, axis=1),
    ),
    axis=1,
)


final_val_X = np.concatenate(
    (
        tox21_valid_dataset.X,
        val_embeddings.numpy(),
        np.expand_dims(graph_tox21_valid_dataset.X, axis=1),
    ),
    axis=1,
)


final_test_X = np.concatenate(
    (
        tox21_test_dataset.X,
        test_embeddings.numpy(),
        np.expand_dims(graph_tox21_test_dataset.X, axis=1),
    ),
    axis=1,
)


train_data_dir = (
    "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/train_dataset"
)
os.chdir(train_data_dir)


train_dataset = dc.data.DiskDataset.from_numpy(
    X=final_train_X,
    y=graph_tox21_train_dataset.y,
    w=graph_tox21_train_dataset.w,
    ids=graph_tox21_train_dataset.ids,
    tasks=tox21_tasks,
    data_dir=train_data_dir,
)

val_data_dir = (
    "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/val_dataset"
)
os.chdir(val_data_dir)
os.getcwd()

val_dataset = dc.data.DiskDataset.from_numpy(
    X=final_val_X,
    y=graph_tox21_valid_dataset.y,
    w=graph_tox21_valid_dataset.w,
    ids=graph_tox21_valid_dataset.ids,
    tasks=tox21_tasks,
    data_dir=val_data_dir,
)

test_data_dir = (
    "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/test_dataset"
)
os.chdir(test_data_dir)
os.getcwd()

test_dataset = dc.data.DiskDataset.from_numpy(
    X=final_test_X,
    y=graph_tox21_test_dataset.y,
    w=graph_tox21_test_dataset.w,
    ids=graph_tox21_test_dataset.ids,
    tasks=tox21_tasks,
    data_dir=test_data_dir,
)
