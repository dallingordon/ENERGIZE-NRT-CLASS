"""Shared helpers for the contrastive Raman-matching cache pipeline (build_cache / score / postprocess).
Wraps the upstream repo (lyn1874/raman_spectra_matching_with_contrastive_learning) without modifying it."""
import hashlib
import json
import os
import sys

import numpy as np

DEFAULT_REPO = ("/projectnb/me500jc/students/dgordon/raman_matching/"
                "raman_spectra_matching_with_contrastive_learning")
DEFAULT_CACHE = "/projectnb/me500jc/students/dgordon/raman_matching/cache"
# grid each model was trained on (grid = np.arange(max_wave)[min_wave:]).
# const.give_args_test() overwrites excellent_unoriented's max_wave with 1024 -> set explicitly.
GRIDS = {"raw": (40, 1496), "excellent_unoriented": (76, 1404)}
# architecture hyper-params hard-coded in upstream test.get_model_baseon_modeldir
ARCH = dict(distance_aggregate="wave_channel_dot_L1", stem_kernel=21, stem_max_dim=64, depth=128)


def enter_repo(repo):
    """Upstream code uses relative paths + package imports: run from its root."""
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def model_paths(model):
    root = "../data/experiments/exp_time/%s/" % model
    versions = sorted(v for v in os.listdir(root) if "version_" in v)
    if len(versions) != 1:
        raise RuntimeError("expected one version_* dir in %s, found %s" % (root, versions))
    model_dir = root + versions[0]
    repeats = sorted(v for v in os.listdir(model_dir) if v.startswith("repeat_"))
    return model_dir, versions[0], repeats


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def cache_dir(cache_root, model, version):
    return os.path.join(cache_root, model, version)


def load_meta(cdir):
    with open(os.path.join(cdir, "meta.json")) as f:
        return json.load(f)


def build_model(test_mod, n_wave, repeat_dir):
    """Same construction as upstream test.get_model_baseon_modeldir."""
    return test_mod.get_model_inception(n_wave, ARCH["distance_aggregate"], repeat_dir,
                                        stem_kernel=ARCH["stem_kernel"], stem_max_dim=ARCH["stem_max_dim"],
                                        depth=ARCH["depth"], print_model=False)


def to_tensor(torch, arr, device):
    """Same shaping as upstream Test.dataloader: [N, W] -> [N, 1, W] float32."""
    arr = np.asarray(arr, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(1).to(device)
