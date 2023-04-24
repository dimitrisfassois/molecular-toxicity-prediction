"""Module for data preprocessing."""
import numpy as np
import deepchem as dc
from transformers import AutoTokenizer, TFRobertaModel
import tensorflow as tf


# Prepare main dataset, tox21
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


# Prepare sider_dataset for transfer learning
sider_tasks, sider_datasets, sider_transformers = dc.molnet.load_sider(
    featurizer="ECFP", reload=False
)
sider_train_dataset, sider_valid_dataset, sider_test_dataset = sider_datasets


sider_tasks, graph_sider_datasets, graph_sider_transformers = dc.molnet.load_sider(
    featurizer="GraphConv", reload=False
)
(
    graph_sider_train_dataset,
    graph_sider_valid_dataset,
    graph_sider_test_dataset,
) = graph_sider_datasets

step = 522
train_size = sider_train_dataset.ids.shape[0]
train_embeddings = []

for i in range(0, train_size, step):
    ids = sider_train_dataset.ids.tolist()[i: i+step]
    tokens = tokenizer(ids,
                            return_tensors="tf",
                            padding=True,
                            truncation=True,
                            max_length=max_len)
    embeddings = roberta(tokens).pooler_output
    train_embeddings.append(embeddings)

train_embeddings = tf.concat(train_embeddings, axis=0)

val_tokens = tokenizer(
    sider_valid_dataset.ids.tolist(),
    return_tensors="tf",
    padding=True,
    truncation=True,
    max_length=max_len,
)
val_embeddings = roberta(val_tokens).pooler_output

test_tokens = tokenizer(
    sider_test_dataset.ids.tolist(),
    return_tensors="tf",
    padding=True,
    truncation=True,
    max_length=max_len,
)
test_embeddings = roberta(test_tokens).pooler_output

final_train_X = np.concatenate(
    (
        sider_train_dataset.X,
        train_embeddings.numpy(),
        np.expand_dims(graph_sider_train_dataset.X, axis=1),
    ),
    axis=1,
)

final_val_X = np.concatenate(
    (
        sider_valid_dataset.X,
        val_embeddings.numpy(),
        np.expand_dims(graph_sider_valid_dataset.X, axis=1),
    ),
    axis=1,
)

final_test_X = np.concatenate(
    (
        sider_test_dataset.X,
        test_embeddings.numpy(),
        np.expand_dims(graph_sider_test_dataset.X, axis=1),
    ),
    axis=1,
)

train_data_dir = (
    "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/sider_dataset/train_dataset"
)
os.chdir(train_data_dir)


train_dataset = dc.data.DiskDataset.from_numpy(
    X=final_train_X,
    y=graph_sider_train_dataset.y,
    w=graph_sider_train_dataset.w,
    ids=graph_sider_train_dataset.ids,
    tasks=sider_tasks,
    data_dir=train_data_dir,
)

val_data_dir = (
    "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/sider_dataset/val_dataset"
)
os.chdir(val_data_dir)
os.getcwd()

val_dataset = dc.data.DiskDataset.from_numpy(
    X=final_val_X,
    y=graph_sider_valid_dataset.y,
    w=graph_sider_valid_dataset.w,
    ids=graph_sider_valid_dataset.ids,
    tasks=sider_tasks,
    data_dir=val_data_dir,
)

test_data_dir = (
    "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/sider_dataset/test_dataset"
)
os.chdir(test_data_dir)
os.getcwd()

test_dataset = dc.data.DiskDataset.from_numpy(
    X=final_test_X,
    y=graph_sider_test_dataset.y,
    w=graph_sider_test_dataset.w,
    ids=graph_sider_test_dataset.ids,
    tasks=sider_tasks,
    data_dir=test_data_dir,
)