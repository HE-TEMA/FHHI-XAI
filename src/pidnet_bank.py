"""Check reference-bank identity and preserve indices when attribution skips rows."""
import hashlib
import json
from pathlib import Path

import numpy as np


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def burnt_artifact_metadata(dataset, checkpoint):
    return {"dataset": dataset.bank_metadata(),
            "checkpoint_sha256": checkpoint_sha256(checkpoint), "class_id": 1}


def burnt_reference_cache(dataset, checkpoint, root):
    """Keep reference crops from different checkpoints/subsets separate."""
    identity = json.dumps(burnt_artifact_metadata(dataset, checkpoint), sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return str(Path(root) / key)


def validate_burnt_artifacts(dataset, checkpoint, crp_dir, pcx_dir, layer):
    expected = burnt_artifact_metadata(dataset, checkpoint)
    for directory in (Path(crp_dir), Path(pcx_dir) / layer):
        path = directory / "burnt_metadata.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; build burnt CRP/PCX artifacts first")
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Burnt artifact configuration mismatch at {path}; rebuild with "
                             "the same checkpoint, dataset order, coverage and preprocessing")


def load_bank_indices(directory, row_count, dataset_length):
    path = Path(directory) / "sample_indices.npy"
    if path.is_file():
        indices = np.load(path, allow_pickle=False)
        if (indices.shape != (row_count,) or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any() or (indices >= dataset_length).any()
                or len(np.unique(indices)) != row_count):
            raise ValueError(f"Invalid bank sample indices in {path}")
        return indices
    if row_count != dataset_length:
        raise ValueError("Bank rows do not match dataset length and sample_indices.npy is missing")
    return np.arange(row_count)
