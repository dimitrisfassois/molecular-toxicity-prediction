import os
import deepchem as dc
import fsspec
import tensorflow as tf
import tensorflow.keras.layers as layers

train_data_dir = "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/sider_dataset/train_dataset"
os.chdir(train_data_dir)

train_dataset = dc.data.DiskDataset(train_data_dir)

val_data_dir = "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/sider_dataset/val_dataset"
os.chdir(val_data_dir)

val_dataset = dc.data.DiskDataset(val_data_dir)

test_data_dir = "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/data/sider_dataset/val_dataset"
os.chdir(test_data_dir)

test_dataset = dc.data.DiskDataset(test_data_dir)

n_tasks = 27
learning_rate = dc.models.optimizers.ExponentialDecay(0.0002, 0.9, 1000)
batch_size = 50
num_heads = key_dim = 0
n_epochs = 30

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

tasks = ['Hepatobiliary disorders',
 'Metabolism and nutrition disorders',
 'Product issues',
 'Eye disorders',
 'Investigations',
 'Musculoskeletal and connective tissue disorders',
 'Gastrointestinal disorders',
 'Social circumstances',
 'Immune system disorders',
 'Reproductive system and breast disorders',
 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
 'General disorders and administration site conditions',
 'Endocrine disorders',
 'Surgical and medical procedures',
 'Vascular disorders',
 'Blood and lymphatic system disorders',
 'Skin and subcutaneous tissue disorders',
 'Congenital, familial and genetic disorders',
 'Infections and infestations',
 'Respiratory, thoracic and mediastinal disorders',
 'Psychiatric disorders',
 'Renal and urinary disorders',
 'Pregnancy, puerperium and perinatal conditions',
 'Ear and labyrinth disorders',
 'Cardiac disorders',
 'Nervous system disorders',
 'Injury, poisoning and procedural complications']

# os.chdir("/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/molecular-toxicity-prediction")

from trainer.model_training import get_model
from data_loaders.data_loaders_utils import get_generator
from models.callbacks import ValidationCallback
from .model_evaluation import evaluate_model

# Train baseline multi-task classifier model from scratch
model_dir = "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/model_checkpoints/transfer_learning/multi_task_classifier/run2"
save_dir = "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/callback_checkpoints/transfer_learning/multi_task_classifier/run2"
model_type = "multi_task_classifier"
dc_model = get_model(
    model_type,
    n_tasks,
    batch_size,
    model_dir,
    learning_rate,
    num_heads,
    key_dim,
)

train_generator = get_generator(
    model_type, train_dataset, batch_size, n_epochs, n_tasks
)


callback = ValidationCallback(
    val_dataset,
    100,
    metric,
    n_tasks,
    model_type,
    batch_size,
    save_dir=save_dir,
)

avg_loss = dc_model.fit_generator(train_generator, callbacks=callback)

dc_model.restore(model_dir=save_dir)

test_generator = get_generator(
    model_type,
    test_dataset,
    batch_size,
    1,
    n_tasks,
    deterministic=True,
)

y_true = test_dataset.y
w = test_dataset.w
y_pred = dc_model.predict_on_generator(test_generator)
y_pred = y_pred[:test_dataset.y.shape[0]]
metric = dc.metrics.roc_auc_score
for i in range(n_tasks):
    score = metric(dc.metrics.to_one_hot(y_true[:, i]), y_pred[:, i], sample_weight=w[:, i])
    print(f"Task: {tasks[i]}, score: {score}")


# Train self-attention model from scratch
model_dir = "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/model_checkpoints/transfer_learning/base_model/run2"
save_dir = "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/callback_checkpoints/transfer_learning/base_model/run2"
model_type = "self_attention"
dc_model = get_model(
    model_type,
    n_tasks,
    batch_size,
    model_dir,
    learning_rate,
    num_heads,
    key_dim,
)

train_generator = get_generator(
    model_type, train_dataset, batch_size, n_epochs, n_tasks
)


callback = ValidationCallback(
    val_dataset,
    100,
    metric,
    n_tasks,
    model_type,
    batch_size,
    save_dir=save_dir,
)


avg_loss = dc_model.fit_generator(train_generator, callbacks=callback)


dc_model.restore(model_dir=save_dir)


test_generator = get_generator(
    model_type,
    test_dataset,
    batch_size,
    1,
    n_tasks,
    deterministic=True,
)


y_true = test_dataset.y
w = test_dataset.w
y_pred = dc_model.predict_on_generator(test_generator)
y_pred = y_pred[:test_dataset.y.shape[0]]
metric = dc.metrics.roc_auc_score
for i in range(n_tasks):
    score = metric(dc.metrics.to_one_hot(y_true[:, i]), y_pred[:, i], sample_weight=w[:, i])
    print(f"Task: {tasks[i]}, score: {score}")


# Use pre-trained self-attention model for transfer learning
model_dir = "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/model_checkpoints/transfer_learning/pretrained_model/run2"
save_dir = "/Users/demetriosfassois/Documents/Columbia/EECSE6895/Project/callback_checkpoints/transfer_learning/pretrained_model/run2"
model_type = "self_attention"
dc_model = get_model(
    model_type,
    n_tasks,
    batch_size,
    model_dir,
    learning_rate,
    num_heads,
    key_dim,
)

# Restore checkpoint from GS bucket
project_id = "molecular-toxicity-prediction"
fs = fsspec.filesystem('gs', project=project_id)
latest_checkpoint = tf.train.latest_checkpoint("gs://molecular-toxicity-prediction/callback_checkpoints/self_attention/run2")
dc_model._ensure_built()
dc_model._checkpoint.restore(latest_checkpoint)

# Swap out the output layer
dc_model.model.output_layer = layers.Dense(n_tasks*2)
dc_model.model.logits = layers.Reshape((n_tasks, 2))

train_generator = get_generator(
    model_type, train_dataset, batch_size, n_epochs, n_tasks
)

callback = ValidationCallback(
    val_dataset,
    100,
    metric,
    n_tasks,
    model_type,
    batch_size,
    save_dir=save_dir,
)

avg_loss = dc_model.fit_generator(train_generator, callbacks=callback)

dc_model.restore(model_dir=save_dir)

test_generator = get_generator(
    model_type,
    test_dataset,
    batch_size,
    1,
    n_tasks,
    deterministic=True,
)

y_true = test_dataset.y
w = test_dataset.w
y_pred = dc_model.predict_on_generator(test_generator)
y_pred = y_pred[:test_dataset.y.shape[0]]
metric = dc.metrics.roc_auc_score
for i in range(n_tasks):
    score = metric(dc.metrics.to_one_hot(y_true[:, i]), y_pred[:, i], sample_weight=w[:, i])
    print(f"Task: {tasks[i]}, score: {score}")


