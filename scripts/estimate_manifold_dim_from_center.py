import argparse
import time
import typing as tp
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import skdim
from joblib import Parallel, delayed
from tqdm import tqdm

args_for_augs = {
    "gain": {"gains": np.linspace(-10, 10, 101, endpoint=True)},
    "time_stretching": {
        "ratios": np.exp(
            np.linspace(
                np.log(0.5),
                np.log(2.0),
                101,
                endpoint=True,
            )
        )
    },
    "pitch_shifting": {"n_steps": np.linspace(-12, 12, 101, endpoint=True)},
}

ap = argparse.ArgumentParser()
ap.add_argument(
    "--aug",
    type=str,
    choices=list(args_for_augs.keys()),
    required=True,
    help="The type of augmentation to visualize.",
)
args = ap.parse_args()
# When true: class means are used to compute scatter matrices
# When false: the embedding of the unaugmented audio is used as the class mean
AUG = args.aug
min_dist_from_center = 6  # Minimum number of augmentations from the center to consider for dimension estimation
ESTIMATOR_CLS = skdim.id.ESS
ESTIMATOR_NAME = ESTIMATOR_CLS.__name__

if Path(f"/home/at4219/scratch/BSD10k_PANN_{AUG}.h5").exists():
    embedding_file = Path(f"/home/at4219/scratch/BSD10k_PANN_{AUG}.h5")
else:
    embedding_file = Path(
        f"/Users/aramis/Heap/computed_embeddings/BSD10k_PANN_{AUG}.h5"
    )
# See which system we're on
if Path("/home/at4219/scratch/").exists():
    data_dir = Path(
        f"/home/at4219/scratch/covs/{AUG}_{ESTIMATOR_NAME}_intrinsic_dimensionality"
    )
else:
    data_dir = Path(
        f"/Users/aramis/covs/{AUG}_{ESTIMATOR_NAME}_intrinsic_dimensionality"
    )


data_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(embedding_file, "r") as hf:
    emb: np.ndarray = hf["embedding"][
        :
    ].squeeze()  # shape (num_files, num_augs, embedding_dim)
print("Loaded embeddings shape: ", emb.shape)
num_classes, num_augs, embedding_dim = emb.shape

id_chart = np.zeros(
    (num_augs // 2 - min_dist_from_center, num_classes), dtype=np.float64
)


def job(aug_subset):
    # Given a bunch of samples (num_samples, num_augs, embedding_dim), compute the intrinsic dimensionality
    # of the manifold for each sample and returns an array of shape (num_samples,)
    output = np.zeros(aug_subset.shape[0], dtype=np.float64)
    for j, sample in enumerate(aug_subset):
        intrinsic_dim = ESTIMATOR_CLS().fit(sample).dimension_
        output[j] = intrinsic_dim
    return output


def arg_generator():
    for dist_from_center in range(min_dist_from_center, num_augs // 2):
        subset = emb[
            :,
            num_augs // 2 - dist_from_center : num_augs // 2 + dist_from_center + 1,
            :,
        ]
        yield subset


# Also save the magnitude of the perturbations
dists_from_center = np.array(list(range(min_dist_from_center, num_augs // 2)))
aug_param_values = args_for_augs[AUG][list(args_for_augs[AUG].keys())[0]]
perturb_magnitudes = aug_param_values[dists_from_center + num_augs // 2]

results = Parallel(n_jobs=-1, return_as="generator")(
    delayed(job)(subset) for subset in arg_generator()
)
for i, result in enumerate(tqdm(results, total=num_augs // 2 - min_dist_from_center)):
    id_chart[i] = result


np.savez(
    data_dir / "intrinsic_dimensionality.npz",
    id_chart=id_chart,
    perturbation_strength=perturb_magnitudes,
)
