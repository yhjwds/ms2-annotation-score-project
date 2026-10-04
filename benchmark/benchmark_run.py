"""
benchmark_run.py
Run benchmark_scoring over the whole benchmark set (3,006 features) and write the
per-candidate score table that benchmark_evaluate.ipynb reads.

    python benchmark_run.py                 full run
    python benchmark_run.py --limit 50      quick run on the first 50 features
    python benchmark_run.py --resume        continue an interrupted run
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import statistics
import sys
import time

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_scoring as bf

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "benchmark")
SPECTRA = os.path.join(DATA, "MS2_spectra_only.json")
ANNOT_DIR = os.path.join(DATA, "level1_annotations")
OUT_DIR = os.path.join(HERE, "results")


TOP_N_PEAKS = 200


PEAK_DISAGREE_PPM = 1.0
CHECKPOINT_EVERY = 200
PER_PEAK_TOP_K = 3

SETTINGS_FOR_RECORD = dict(
    ppm=bf.TOL_PPM, min_da=bf.TOL_MIN,
    method="FragmentOnBonds", max_breaks=bf.MAX_BREAKS,
    fragment_fallback=bf.FRAGMENT_FALLBACK,
    min_rel_int=bf.MIN_REL_INT, drop_above_precursor=bf.DROP_ABOVE_PRECURSOR,
    top_n_peaks=TOP_N_PEAKS, include_generic=False,
    salt_policy="skip", peak_disagree_ppm=PEAK_DISAGREE_PPM,
    n_loss_rules=len(bf._LOSS_RULES), n_universal_losses=len(bf._UNIVERSAL),
    fragment_adducts_pos=bf.FRAGMENT_ADDUCTS_POS,
    fragment_adducts_neg=bf.FRAGMENT_ADDUCTS_NEG,
)

METHOD = "FragmentOnBonds"


def load_spectra(path=SPECTRA):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_annotations(feature_id, ann_dir=ANNOT_DIR):
    """One feature's candidate list, with `id` = its ROW NUMBER in the file. That row"""
    path = os.path.join(ann_dir, f"{feature_id}_MS1annotation.csv")
    ann = pd.read_csv(path)
    return ann.assign(id=[str(i) for i in range(len(ann))])


def candidate_status(smiles, adduct, lut):
    """Why a candidate can or cannot be scored. Skipped candidates stay in the output"""
    s = str(smiles)
    if "." in s:


        return "skipped_salt"
    if Chem.MolFromSmiles(s) is None:
        return "skipped_smiles"
    if str(adduct) not in lut:
        return "skipped_adduct"
    return "scored"


def recover_precursor(ann, lut):
    """(precursor m/z, disagreement in ppm, n candidates used)."""
    vals = []
    for _, r in ann.iterrows():
        if candidate_status(r["smiles"], r["adduct"], lut) != "scored":
            continue
        mol = Chem.MolFromSmiles(str(r["smiles"]))
        vals.append(bf.adduct_mz(ExactMolWt(mol), lut[str(r["adduct"])])
                    * (1 + float(r["ppm"]) / 1e6))
    if not vals:
        return None, None, 0
    med = statistics.median(vals)
    spread = (max(vals) - min(vals)) / med * 1e6 if len(vals) > 1 else 0.0
    return med, spread, len(vals)


def prepare_peaks(raw, precursor):
    """Raw [[m/z, intensity], ...] -> the peak list the method sees, plus what the"""
    n_raw = len(raw)
    top = sorted(raw, key=lambda p: -p[1])[:TOP_N_PEAKS]
    total_int = sum(p[1] for p in raw) or 1.0
    kept_int = sum(p[1] for p in top)
    mx = max((p[1] for p in top), default=1.0) or 1.0
    peaks = [(float(mz), 100.0 * float(i) / mx) for mz, i in top]
    peaks = [(mz, ri) for mz, ri in peaks
             if ri >= bf.MIN_REL_INT and (precursor is None or mz < precursor)]
    peaks.sort()
    return peaks, dict(n_peaks_raw=n_raw, n_peaks_kept=len(peaks),
                       intensity_kept_pct=round(100.0 * kept_int / total_int, 2))


def run_feature(fid, spectrum, lut):
    """-> (submission rows, feature summary, per-peak rows). Never raises: a feature"""
    ann = read_annotations(fid)
    ionisation = str(ann["ionisation"].iloc[0]) if len(ann) else ""
    status = [candidate_status(r["smiles"], r["adduct"], lut)
              for _, r in ann.iterrows()]
    scored_ann = ann[[s == "scored" for s in status]]

    summary = dict(feature_id=fid, ionisation=ionisation, n_candidates=len(ann),
                   n_scored=len(scored_ann), n_skipped=len(ann) - len(scored_ann))

    def _rows(which, st_of):
        """Rows for candidates we are NOT returning a score for. They stay in the"""
        return [dict(feature_id=fid, id=r["id"], name=r["name"], smiles=r["smiles"],
                     adduct=r["adduct"], status=st_of(st))
                for (_, r), st in zip(which.iterrows(), status) if st_of(st)]

    skipped_rows = _rows(ann, lambda st: st if st != "scored" else None)

    def _all_unscored(reason):
        """Every candidate of a feature we could not analyse at all."""
        return [dict(feature_id=fid, id=r["id"], name=r["name"], smiles=r["smiles"],
                     adduct=r["adduct"], status=(st if st != "scored" else reason))
                for (_, r), st in zip(ann.iterrows(), status)]

    if not len(scored_ann):
        summary.update(feature_status="no_scorable_candidate")
        return skipped_rows, summary, []

    prec, spread, _ = recover_precursor(scored_ann, lut)
    peaks, pstats = prepare_peaks(spectrum, prec)
    summary.update(precursor_mz=round(prec, 5) if prec else None,
                   precursor_disagree_ppm=round(spread, 3) if spread is not None else None,
                   precursor_disagree=bool(spread and spread > PEAK_DISAGREE_PPM),
                   **pstats)
    if not peaks:
        summary.update(feature_status="no_peaks")
        return _all_unscored("no_peaks"), summary, []

    try:
        cands = bf.analyze_candidates(scored_ann, peaks, prec, method=METHOD,
                                      max_breaks=bf.MAX_BREAKS)
    except Exception as exc:
        summary.update(feature_status=f"error:{type(exc).__name__}", error=str(exc)[:200])
        return _all_unscored(f"error:{type(exc).__name__}"), summary, []

    sub = bf.submission_table(cands, feature_id=fid)
    sub["status"] = "scored"
    smi = dict(zip(ann["id"].astype(str), ann["smiles"]))
    sub["smiles"] = sub["id"].astype(str).map(smi)

    best = cands[0]
    parts = best.get("score_parts", {})
    summary.update(
        feature_status="all_zero" if bool(sub["all_zero"].iloc[0]) else "ok",
        all_zero=bool(sub["all_zero"].iloc[0]),
        best_name=best["name"], best_score=round(float(best["score"]), 4),
        best_adduct=best["adduct"], best_via=best.get("via", ""),


        n_tied_at_top=int((sub["score"] == sub["score"].max()).sum()) if len(sub) else 0,

        tiebreak_resolves=bool(len(sub) and (sub["rank_tiebreak"] == 1).sum() == 1),
        task1_pct=parts.get("task1_pct"), task2_diagnostic_pct=parts.get("task2_diagnostic_pct"),
        task2_generic_pct=parts.get("task2_generic_pct"), unexplained_pct=parts.get("unexplained_pct"),
        n_fallback=sum(1 for c in cands if c.get("via") == "exhaustive-fallback"),
    )

    per_peak = []
    for c in cands[:PER_PEAK_TOP_K]:
        try:
            t = bf.explained_table(c, c["adduct"])
        except Exception:
            continue
        t.insert(0, "cand_name", c["name"])
        t.insert(0, "cand_id", c["id"])
        t.insert(0, "feature_id", fid)
        per_peak.append(t)
    per_peak = pd.concat(per_peak, ignore_index=True) if per_peak else None
    return (sub.to_dict("records") + skipped_rows, summary,
            [] if per_peak is None else per_peak.to_dict("records"))


SUB_COLS = ["feature_id", "id", "name", "smiles", "adduct", "status", "score",
            "n_theoretical", "n_matched", "n_explained", "n_bonds_broken", "via",
            "rank_best", "rank_worst", "n_tied", "identical_ions",
            "rank_tiebreak", "all_zero"]


def _write(rows, path, cols=None):
    df = pd.DataFrame(rows)
    if cols:
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(path, index=False, encoding="utf-8-sig",
              compression="gzip" if path.endswith(".gz") else None)
    return len(df)


def main(limit=None, resume=False, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    spectra = load_spectra()
    lut = bf.all_adducts()
    ids = sorted(spectra)
    if limit:
        ids = ids[:limit]

    done_path = os.path.join(out_dir, "_done.txt")
    done = set()
    if resume and os.path.exists(done_path):
        done = set(open(done_path, encoding="utf-8").read().split())
        print(f"[resume] {len(done)} features already finished, skipping them")
    todo = [i for i in ids if i not in done]

    sub_rows, summaries, peak_rows = [], [], []


    ck = max([int(f[3:6]) for f in os.listdir(out_dir)
              if f.startswith("_ck") and f.endswith("_submission.csv")], default=0)
    print(f"[start] {len(todo)} features | ppm={bf.TOL_PPM} min_da={bf.TOL_MIN} "
          f"method={METHOD} max_breaks={bf.MAX_BREAKS} top_n={TOP_N_PEAKS}")
    for n, fid in enumerate(todo, 1):
        try:
            s, summary, pp = run_feature(fid, spectra[fid], lut)
        except Exception as exc:
            s, summary, pp = [], dict(feature_id=fid,
                                      feature_status=f"fatal:{type(exc).__name__}",
                                      error=str(exc)[:200]), []
        sub_rows += s
        summaries.append(summary)
        peak_rows += pp
        done.add(fid)
        if n % 50 == 0 or n == len(todo):
            el = time.perf_counter() - t0
            eta = el / n * (len(todo) - n)
            print(f"  {n}/{len(todo)}  已用 {el/60:.1f} min  预计剩余 {eta/60:.1f} min",
                  flush=True)
        if n % CHECKPOINT_EVERY == 0:
            ck += 1
            _write(sub_rows, os.path.join(out_dir, f"_ck{ck:03d}_submission.csv"), SUB_COLS)
            _write(summaries, os.path.join(out_dir, f"_ck{ck:03d}_summary.csv"))
            _write(peak_rows, os.path.join(out_dir, f"_ck{ck:03d}_peaks.csv.gz"))
            with open(done_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(sorted(done)))
            sub_rows, summaries, peak_rows = [], [], []


    def _merge(pattern, final, cols=None):
        parts = [pd.read_csv(os.path.join(out_dir, f))
                 for f in sorted(os.listdir(out_dir)) if f.startswith("_ck") and f.endswith(pattern)]
        return parts, final, cols

    for pattern, tail_rows, final, cols in (
            ("_submission.csv", sub_rows, "submission_ours.csv", SUB_COLS),
            ("_summary.csv", summaries, "per_feature_summary.csv", None),
            ("_peaks.csv.gz", peak_rows, "per_peak_long.csv.gz", None)):
        parts = [pd.read_csv(os.path.join(out_dir, f))
                 for f in sorted(os.listdir(out_dir))
                 if f.startswith("_ck") and f.endswith(pattern)]
        if tail_rows:
            parts.append(pd.DataFrame(tail_rows))
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        sort_key = [c for c in ("feature_id", "rank_tiebreak", "id") if c in df.columns]
        if sort_key:
            df = df.sort_values(sort_key, kind="mergesort")
        n = _write(df.to_dict("records"), os.path.join(out_dir, final), cols)
        print(f"[write] {final}: {n} 行")

    with open(os.path.join(out_dir, "run_settings.json"), "w", encoding="utf-8") as fh:
        rec = dict(SETTINGS_FOR_RECORD)
        rec["per_peak_top_k"] = PER_PEAK_TOP_K
        rec["n_features"] = len(ids)
        rec["run_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["runtime_min"] = round((time.perf_counter() - t0) / 60, 2)
        mod = os.path.join(HERE, "benchmark_scoring.py")
        rec["module_sha256"] = hashlib.sha256(open(mod, "rb").read()).hexdigest()
        json.dump(rec, fh, indent=2, ensure_ascii=False)
    print(f"[done] {(time.perf_counter()-t0)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个特征")
    ap.add_argument("--resume", action="store_true", help="跳过 _done.txt 里已完成的")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    main(limit=a.limit, resume=a.resume, out_dir=a.out)
