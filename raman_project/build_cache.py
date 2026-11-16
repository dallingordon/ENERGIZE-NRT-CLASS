"""Stage 3a: encode the RRUFF reference set once per model and cache everything.

For model M (raw | excellent_unoriented) writes  <cache>/<M>/<version>/:
  refs.npz              reference spectra on the model grid [N_ref, W], wavenumber grid, label ids,
                        mineral name per reference, RRUFF filename per reference, label_name table
  repeat_k_features.pt  reference features from repeat k's encoder (model.forward_test, as upstream Test)
  meta.json             model/version, grid, arch params, per-repeat checkpoint path + size + mtime + sha256,
                        torch/numpy versions, model.training flag, feature shapes, timings

Run on a compute node (see build_cache.qsub). Usage:
    python build_cache.py --model excellent_unoriented
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmatch_common as rc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(rc.GRIDS))
    ap.add_argument("--repo", default=rc.DEFAULT_REPO)
    ap.add_argument("--cache", default=rc.DEFAULT_CACHE)
    ap.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    a = ap.parse_args()

    rc.enter_repo(a.repo)
    import torch
    import const
    import test
    if os.environ.get("NSLOTS"):
        torch.set_num_threads(int(os.environ["NSLOTS"]))

    model_dir, version, repeats = rc.model_paths(a.model)
    cdir = rc.cache_dir(a.cache, a.model, version)
    if os.path.exists(os.path.join(cdir, "meta.json")) and not a.force:
        sys.exit("cache exists: %s  (use --force to rebuild)" % cdir)
    os.makedirs(cdir, exist_ok=True)

    # --- references, exactly as test_experiment.ipynb ---
    args = const.give_args_test(raman_type=a.model)
    args["min_wave"], args["max_wave"] = rc.GRIDS[a.model]
    args["pre_define_tt_filenames"] = True
    dir2load_data = model_dir + "/data_splitting/"
    t0 = time.time()
    tr_data, _tt, _, label_name = test.get_data(args, dir2load_data, "cls", False, "../data/public_dataset/")
    tr_spectrum = np.asarray(tr_data[0], dtype=np.float32)
    tr_label = np.asarray(tr_data[1]).astype(np.int32)
    label_name = np.asarray(label_name)
    n_ref, n_wave = tr_spectrum.shape
    wavenumber = np.arange(args["max_wave"])[args["min_wave"]:]
    assert len(wavenumber) == n_wave, (len(wavenumber), n_wave)

    fn_path = dir2load_data + "tr_filename.npy"
    tr_filename = np.array(["ref%d" % i for i in range(n_ref)])
    fn_note = "tr_filename.npy missing"
    if os.path.exists(fn_path):
        fn = np.load(fn_path, allow_pickle=True)
        if len(fn) == n_ref:
            tr_filename, fn_note = np.asarray(fn).astype(str), "from tr_filename.npy"
        else:
            fn_note = "tr_filename.npy has %d entries vs %d refs; using ref indices" % (len(fn), n_ref)
    print("references: %d spectra, %d minerals, %d wavenumbers (%s) [%.0fs]"
          % (n_ref, len(np.unique(tr_label)), n_wave, fn_note, time.time() - t0))

    np.savez_compressed(os.path.join(cdir, "refs.npz"), tr_spectrum=tr_spectrum, tr_label=tr_label,
                        tr_mineral=label_name[tr_label].astype(str), tr_filename=tr_filename,
                        label_name=label_name.astype(str), wavenumber=wavenumber)

    meta = dict(model=a.model, version=version, model_dir=os.path.abspath(model_dir), repo=a.repo,
                min_wave=args["min_wave"], max_wave=args["max_wave"], n_wave=int(n_wave),
                norm_method=args.get("norm_method"), arch=rc.ARCH, n_ref=int(n_ref),
                n_minerals=int(len(np.unique(tr_label))), filenames=fn_note,
                torch=torch.__version__, numpy=np.__version__, device=str(test.device),
                created=datetime.datetime.now().isoformat(timespec="seconds"), repeats={})

    x = rc.to_tensor(torch, tr_spectrum, test.device)
    for r in repeats:
        rdir = model_dir + "/" + r
        ckpt = [rdir + "/" + v for v in sorted(os.listdir(rdir)) if ".pt" in v][-1]  # as upstream
        t0 = time.time()
        model = rc.build_model(test, n_wave, rdir)
        with torch.no_grad():
            # upstream Test.__init__: encodes ALL references in a single batch
            feats = model.forward_test(x, [])
        if isinstance(feats, (list, tuple)):
            raise RuntimeError("forward_test returned %s, expected a tensor" % type(feats))
        torch.save(feats.cpu(), os.path.join(cdir, "%s_features.pt" % r))
        st = os.stat(ckpt)
        meta["repeats"][r] = dict(ckpt=os.path.abspath(ckpt), ckpt_bytes=st.st_size, ckpt_mtime=st.st_mtime,
                                  ckpt_sha256=rc.sha256(ckpt), model_training_flag=bool(model.training),
                                  feature_shape=list(feats.shape), feature_dtype=str(feats.dtype),
                                  seconds=round(time.time() - t0, 1))
        print("  %s: features %s  training=%s  [%.0fs]" % (r, tuple(feats.shape), model.training, time.time() - t0))
        del model, feats

    with open(os.path.join(cdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("cache written:", cdir)


if __name__ == "__main__":
    main()
