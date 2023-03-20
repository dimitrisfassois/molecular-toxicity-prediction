"""Data loaders for data with language features."""
import numpy as np
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
import deepchem as dc

from util.constants import CONST

def get_disk_dataset(fs, data_dir):
    return dc.data.datasets.DiskDataset.from_numpy(
        X=np.load(fs.open(f"{data_dir}/shard-0-X.npy"), allow_pickle=True),
        y=np.load(fs.open(f"{data_dir}/shard-0-y.npy"), allow_pickle=True),
        w=np.load(fs.open(f"{data_dir}/shard-0-w.npy"), allow_pickle=True),
        ids=np.load(fs.open(f"{data_dir}/shard-0-ids.npy"), allow_pickle=True),
        tasks=CONST.TASKS
    )



def full_dataset_generator(dataset, batch_size, n_epochs, n_tasks, deterministic):
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(
        dataset.iterbatches(batch_size, n_epochs, deterministic=deterministic, pad_batches=True)
    ):
        multiConvMol = ConvMol.agglomerate_mols(X_b[:, -1])
        graph_inputs = [
            multiConvMol.get_atom_features(),
            multiConvMol.deg_slice,
            np.array(multiConvMol.membership),
        ]
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
            graph_inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
        fc_net_inputs = [X_b[:, :-1].astype("float32")]
        labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
        weights = [w_b]
        yield fc_net_inputs + graph_inputs, labels, weights


def fingerprints_generator(dataset, batch_size, n_epochs, n_tasks, deterministic):
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(
        dataset.iterbatches(batch_size, n_epochs, deterministic=deterministic, pad_batches=True)
    ):
        fc_net_inputs = [X_b[:, :1024].astype("float32")]
        labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
        weights = [w_b]
        yield fc_net_inputs, labels, weights
