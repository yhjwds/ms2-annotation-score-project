"""
benchmark_scoring.py
Scoring method used for the 3,006-spectrum benchmark (run through benchmark_run.py).

Same pipeline as ms2_scoring.py, with the settings frozen for the benchmark:
5 ppm / 0.001 Da tolerance (as in MetFrag), FragmentOnBonds with up to 2 broken bonds,
a fallback to all acyclic single bonds when BRICS gives no fragment, a larger
substructure-gated neutral-loss library, and a tie-break column for equal scores.
"""
import os
import re
import math
import collections
from itertools import combinations
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog("rdApp.*")

PROTON, ELECTRON = 1.0072765, 0.0005486
TOL_PPM, TOL_MIN = 5.0, 0.001


DEFAULT_METHOD = "FragmentOnBonds"


MAX_BREAKS = 2


MIN_REL_INT = 1.0


DROP_ABOVE_PRECURSOR = True


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "example_features")


def set_tolerance(ppm, min_da=None):
    """Set the m/z matching tolerance used by ALL peak matching (Task 1 & 2)."""
    global TOL_PPM, TOL_MIN
    TOL_PPM = float(ppm)
    if min_da is not None:
        TOL_MIN = float(min_da)
    return TOL_PPM, TOL_MIN


ADDUCTS_URL = "https://raw.githubusercontent.com/francescodc87/ipaPy2/main/DB/adducts.csv"
ADDUCTS_LOCAL = os.path.join(HERE, "..", "data", "adducts.csv")


FRAGMENT_ADDUCTS_POS = ["M+H", "M+", "M+Na", "M+2H", "2M+H", "M+NH4", "M+K"]
FRAGMENT_ADDUCTS_NEG = ["M-H", "M+Cl"]
FRAGMENT_ADDUCTS = FRAGMENT_ADDUCTS_POS
_ADDUCT_CACHE = {}
_ADDUCT_ROWS = None


def load_adducts(names=None, url=ADDUCTS_URL, local=ADDUCTS_LOCAL):
    """Load (name, mult, charge, mass) for the selected adducts from ipaPy2's"""
    names = names or FRAGMENT_ADDUCTS
    try:
        df = pd.read_csv(url)
        src = "web (github raw)"
    except Exception as e:
        df = pd.read_csv(local)
        src = f"local copy ({os.path.basename(local)})"
    df = df[df["name"].isin(names)]
    order = {n: i for i, n in enumerate(names)}
    df = df.sort_values("name", key=lambda s: s.map(order))
    out = [dict(name=r["name"], mult=int(r["Mult"]), charge=int(r["Charge"]),
                mass=float(r["Mass"]), mode=str(r["Ion_mode"]).strip().lower())
           for _, r in df.iterrows()]
    missing = [n for n in names if n not in set(df["name"])]
    note = f"  (missing: {', '.join(missing)})" if missing else ""
    print(f"[adducts] {len(out)} loaded from {src}: "
          f"{', '.join(a['name'] for a in out)}{note}")
    return out


def all_adducts():
    """Every row of adducts.csv as {name -> dict}, loaded once. Used to look up the"""
    global _ADDUCT_ROWS
    if _ADDUCT_ROWS is None:
        try:
            df = pd.read_csv(ADDUCTS_URL)
        except Exception:
            df = pd.read_csv(ADDUCTS_LOCAL)
        _ADDUCT_ROWS = {}
        for _, r in df.iterrows():
            _ADDUCT_ROWS.setdefault(str(r["name"]), dict(
                name=str(r["name"]), mult=int(r["Mult"]), charge=int(r["Charge"]),
                mass=float(r["Mass"]), mode=str(r["Ion_mode"]).strip().lower()))
    return _ADDUCT_ROWS


def adduct_mode(adduct_name, default="positive"):
    """'positive' / 'negative' for a precursor adduct name ('M+H' -> positive,"""
    a = all_adducts().get(str(adduct_name or "").strip())
    return a["mode"] if a else default


def get_fragment_adducts(mode="positive"):
    """The fragment-projection adducts for one polarity, loaded once per mode."""
    if mode not in _ADDUCT_CACHE:
        names = FRAGMENT_ADDUCTS_NEG if mode == "negative" else FRAGMENT_ADDUCTS_POS
        _ADDUCT_CACHE[mode] = load_adducts(names)
    return _ADDUCT_CACHE[mode]


def adduct_mz(neutral_mass, adduct):
    """Theoretical ion m/z of a neutral mass under one adduct dict:"""
    return (adduct["mult"] * neutral_mass) / abs(adduct["charge"]) + adduct["mass"]


def cap(frag_smiles):
    """Cap BRICS dummy atoms with H (so a fragment becomes a real molecule)."""
    return Chem.MolFromSmiles(re.sub(r"\[\d*\*\]|\*", "[H]", frag_smiles))


FRAGMENT_FALLBACK = True
CACHE_FRAGMENTS = True
_FRAG_CACHE, _FMAP_CACHE = {}, {}


def _frag_key(mol, method, max_breaks):
    """Cache key for a fragmentation: the canonical SMILES plus the settings that"""
    return (Chem.MolToSmiles(mol), method, max_breaks, FRAGMENT_FALLBACK)


def clear_fragment_cache():
    """Drop both caches. Call this after changing FRAGMENT_FALLBACK or monkey-patching"""
    _FRAG_CACHE.clear()
    _FMAP_CACHE.clear()


def _bond_set(mol, selector="brics"):
    """The bonds we are allowed to cut, as bond indices. This is the ONLY place that"""
    if selector == "brics":
        return [mol.GetBondBetweenAtoms(a, b).GetIdx()
                for (a, b), _ in BRICS.FindBRICSBonds(mol)]
    if selector == "acyclic-single":
        return [b.GetIdx() for b in mol.GetBonds()
                if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE]
    raise ValueError(f"unknown bond selector: {selector}")


def _pieces_from(mol, bonds, max_breaks):
    """Break every combination of 1..max_breaks of the given bonds -> dummy-labelled"""
    pieces = set()
    for n in range(1, max_breaks + 1):
        for combo in combinations(bonds, n):
            fm = Chem.FragmentOnBonds(mol, combo, addDummies=True)
            pieces.update(Chem.MolToSmiles(fm).split("."))
    return pieces


def _combinatorial_pieces(mol, max_breaks, fallback=None):
    """(pieces, bonds, via) - the fragmentation, plus WHICH bonds produced it."""
    if fallback is None:
        fallback = FRAGMENT_FALLBACK
    bonds = _bond_set(mol, "brics")
    pieces = _pieces_from(mol, bonds, max_breaks)
    if fallback and len(pieces) <= 1:
        alt = _bond_set(mol, "acyclic-single")
        if alt:
            return _pieces_from(mol, alt, max_breaks), alt, "exhaustive-fallback"
    return pieces, bonds, "brics"


def brics_cut(mol, method=DEFAULT_METHOD, max_breaks=MAX_BREAKS, fallback=None):
    """Return the raw fragment SMILES (with dummy atoms) for the chosen method."""
    if method == "BRICSDecompose":
        return sorted(BRICS.BRICSDecompose(mol))
    if method == "FragmentOnBonds":
        return sorted(_combinatorial_pieces(mol, max_breaks, fallback)[0])
    raise ValueError(f"unknown method: {method}")


def brics_fragments(mol, method=DEFAULT_METHOD, max_breaks=MAX_BREAKS, fallback=None):
    """Cut -> cap -> mass."""
    ck = _frag_key(mol, method, max_breaks) if CACHE_FRAGMENTS else None
    if ck is not None and ck in _FRAG_CACHE:
        return _FRAG_CACHE[ck]
    if method == "FragmentOnBonds":
        pieces, _bonds, via = _combinatorial_pieces(mol, max_breaks, fallback)
        pieces = sorted(pieces)
    else:
        pieces, via = brics_cut(mol, method, max_breaks, fallback), "brics"
    out = {}
    for piece in pieces:
        m = cap(piece)
        if m is None:
            continue
        smi = Chem.MolToSmiles(m)
        out[smi] = dict(smiles=smi,
                        formula=rdMolDescriptors.CalcMolFormula(m),
                        mass=ExactMolWt(m),
                        charge=Chem.GetFormalCharge(m),
                        via=via)
    frags = list(out.values())
    if ck is not None:
        _FRAG_CACHE[ck] = frags
    return frags


def fragment_map(mol, method=DEFAULT_METHOD, max_breaks=MAX_BREAKS):
    """WHERE each fragment came from: canonical capped SMILES ->"""
    ck = _frag_key(mol, method, max_breaks) if CACHE_FRAGMENTS else None
    if ck is not None and ck in _FMAP_CACHE:
        return _FMAP_CACHE[ck]
    _pieces, bonds, _via = _combinatorial_pieces(mol, max_breaks)
    n_at = mol.GetNumAtoms()
    combos = ([tuple(bonds)] if method == "BRICSDecompose" else
              [c for n in range(1, max_breaks + 1) for c in combinations(bonds, n)])
    out = {}
    for combo in combos:
        fm = Chem.FragmentOnBonds(mol, list(combo), addDummies=True)
        mapping = []
        for piece, idxs in zip(Chem.GetMolFrags(fm, asMols=True, sanitizeFrags=False,
                                                fragsMolAtomMapping=mapping), mapping):
            capped = cap(Chem.MolToSmiles(piece))
            if capped is None:
                continue
            smi = Chem.MolToSmiles(capped)
            if smi not in out:
                out[smi] = ([i for i in idxs if i < n_at], list(combo))
    if ck is not None:
        _FMAP_CACHE[ck] = out
    return out


def _bonds_in(mol, atoms):
    """Bond indices with BOTH ends inside `atoms` - the bonds that draw that region."""
    a = set(atoms)
    return [b.GetIdx() for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in a and b.GetEndAtomIdx() in a]


def loss_sites(mol):
    """WHERE a neutral loss sits on the molecule: {loss label -> (atoms, cut bonds)}."""
    out, n_at = {}, mol.GetNumAtoms()
    for cC, _o, oEst, _g in mol.GetSubstructMatches(_ACYL_SMARTS):
        bond = mol.GetBondBetweenAtoms(cC, oEst).GetIdx()
        fm = Chem.FragmentOnBonds(mol, [bond], addDummies=True)
        mapping = []
        for piece, idxs in zip(Chem.GetMolFrags(fm, asMols=True, sanitizeFrags=False,
                                                fragsMolAtomMapping=mapping), mapping):
            smi = Chem.MolToSmiles(piece)
            if "*" not in smi:
                continue
            acid = Chem.MolFromSmiles(re.sub(r"\[\d*\*\]|\*", "O", smi))
            if acid is None or not acid.HasSubstructMatch(_ACID_SMARTS):
                continue
            out.setdefault(_acid_label(acid), ([i for i in idxs if i < n_at], [bond]))
    for patt, losses in _LOSS_RULES:
        hit = mol.GetSubstructMatch(patt) if patt is not None else ()
        for _f, nm in (losses if hit else []):
            out.setdefault(nm, (list(hit), []))
    return out


def fragment_ions(frags, adducts=None, mode="positive"):
    """Neutral fragment -> theoretical ion m/z, one entry per selected adduct."""
    if adducts is None:
        adducts = get_fragment_adducts(mode)
    ions = []
    for f in frags:
        if f["charge"] != 0:


            if mode == "negative":
                mz = f["mass"] - (1 + f["charge"]) * PROTON + ELECTRON
            else:
                mz = f["mass"] + (1 - f["charge"]) * PROTON - ELECTRON
            ions.append(dict(mz=mz, ion_type="intrinsic", adduct="intrinsic",
                             formula=f["formula"], frag=f["smiles"]))
            continue
        for a in adducts:
            ions.append(dict(mz=adduct_mz(f["mass"], a),
                             ion_type=a["name"], adduct=a["name"],
                             formula=f["formula"], frag=f["smiles"]))
    return ions


_NAME_SMARTS = [
    ("[NX4+](C)(C)(C)CCO[PX4]",                     "phosphocholine"),
    ("[NX4+](C)(C)(C)CC[OX2]",                      "choline"),
    ("[NX4+](C)(C)(C)",                             "trimethylammonium"),
    ("[CX4]([OX2]C(=O)[#6])[CX4][OX2]C(=O)[#6]",    "diacylglycerol (DAG)"),
    ("[CX4]([OX2]C(=O)[#6])[CX4][CX4][OX2]C(=O)[#6]", "diacylglycerol (DAG)"),
    ("[CX3](=O)[OX2H1]",                            "fatty acid"),
    ("[PX4](=O)([OX2H1,OX1-])[OX2H1,OX1-]",         "phosphate"),
]
_NAME_QUERIES = [(Chem.MolFromSmarts(s), n) for s, n in _NAME_SMARTS]


def name_fragment(smiles):
    """Heuristic class name for a fragment via SMARTS substructure match."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ""
    for q, nm in _NAME_QUERIES:
        if q is not None and m.HasSubstructMatch(q):
            return nm
    return ""


_C13 = 1.003355
_ISO_TOL = 0.004

_ADDUCTS = [("[M+H]+", PROTON), ("[M+NH4]+", 18.033823),
            ("[M+Na]+", 22.989218), ("[M+K]+", 38.963158)]
_ADDUCTS_NEG = [("[M-H]-", -PROTON), ("[M+Cl]-", 34.969402)]


def _guess_adduct(precursor_mz, neutral_mass, mode="positive"):
    """Name the adduct of the precursor by matching (precursor - neutral) to the"""
    table = _ADDUCTS_NEG if mode == "negative" else _ADDUCTS
    d = precursor_mz - neutral_mass
    label, add = min(table, key=lambda a: abs(d - a[1]))
    return label, neutral_mass + add


def _special_tags(peaks, precursor_mz):
    """Tag each peak that is the surviving precursor ion or a +1 (13C) isotope,"""
    tags = {}
    for mz, ri in peaks:
        if precursor_mz is not None and match(mz, precursor_mz):
            tags[round(mz, 4)] = "precursor"
            continue

        for mz2, ri2 in peaks:
            if abs((mz - mz2) - _C13) <= _ISO_TOL and ri2 >= ri:
                tags[round(mz, 4)] = "isotope"
                break
    return tags


def _annotate_special(rows, mol, precursor_mz, mode="positive"):
    """Give precursor and isotope peaks a real annotation (not just a tag):"""
    neutral = ExactMolWt(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    for r in rows:
        if r["special"] == "precursor" and precursor_mz is not None:
            ion_type, theo = _guess_adduct(precursor_mz, neutral, mode)
            r["name"] = "precursor (intact, unfragmented ion)"
            r["ion_type"] = ion_type
            r["best_adduct"] = ion_type.strip("[]+")
            r["formula"] = formula
            r["theo_mz"] = round(theo, 4)
            r["ppm"] = round((r["exp_mz"] - theo) / r["exp_mz"] * 1e6, 1)
        elif r["special"] == "isotope":

            parent = None
            for p in rows:
                if abs((r["exp_mz"] - p["exp_mz"]) - _C13) <= _ISO_TOL and p["rel_int"] >= r["rel_int"]:
                    if parent is None or p["rel_int"] > parent["rel_int"]:
                        parent = p
            base = f'{parent["exp_mz"]:.3f}' if parent else "a monoisotopic peak"
            extra = f' ({parent["name"]})' if parent and parent["name"] else ""
            r["name"] = f"13C isotope of m/z {base}{extra}"
            if parent:
                r["formula"] = (parent["formula"] + " +[13C]") if parent["formula"] else ""
                r["ion_type"] = parent["ion_type"]
                r["best_adduct"] = parent.get("best_adduct", "")
                r["theo_mz"] = round(parent["exp_mz"] + _C13, 4)
                r["ppm"] = round((r["exp_mz"] - r["theo_mz"]) / r["exp_mz"] * 1e6, 1)


def match(exp_mz, ion_mz):
    """True if an experimental peak and a theoretical ion agree within tolerance."""
    return abs(exp_mz - ion_mz) <= max(TOL_MIN, exp_mz * TOL_PPM * 1e-6)


def analyze(peaks, smiles, method=DEFAULT_METHOD, max_breaks=MAX_BREAKS,
            precursor_mz=None, adduct=None):
    """Full pipeline for ONE candidate: cut -> cap -> mass -> ions -> match peaks."""
    mode = adduct_mode(adduct)
    mol = Chem.MolFromSmiles(smiles)
    frags = brics_fragments(mol, method, max_breaks)
    ions = fragment_ions(frags, mode=mode)
    tags = _special_tags(peaks, precursor_mz)
    rows = []
    for mz, ri in peaks:
        hits = [io for io in ions if match(mz, io["mz"])]


        best = min(hits, key=lambda io: abs(mz - io["mz"])) if hits else None
        hit_adducts = list(dict.fromkeys(io["adduct"] for io in hits))
        alt = [a for a in hit_adducts if best and a != best["adduct"]]
        rows.append(dict(exp_mz=round(mz, 4), rel_int=round(ri, 3),
                         special=tags.get(round(mz, 4), ""),
                         matched=best is not None,
                         name=name_fragment(best["frag"]) if best else "",
                         formula=best["formula"] if best else "",
                         ion_type=best["ion_type"] if best else "",
                         best_adduct=best["adduct"] if best else "",
                         alt_adducts=", ".join(alt),
                         frag_smiles=best["frag"] if best else "",
                         theo_mz=round(best["mz"], 4) if best else None,
                         ppm=round((mz - best["mz"]) / mz * 1e6, 1) if best else None,
                         n_hits=len(hits),
                         n_frag=len({io["frag"] for io in hits})))
    _annotate_special(rows, mol, precursor_mz, mode)
    for r in rows:
        r["kind"] = r["special"] or ("fragment" if r["matched"] else "")
    n = sum(r["matched"] for r in rows)
    total_int = sum(r["rel_int"] for r in rows) or 1
    cov = dict(n_matched=n, n_total=len(rows),
               pct_peaks=round(100 * n / len(rows), 1) if rows else 0,
               pct_intensity=round(sum(r["rel_int"] for r in rows if r["matched"]) /
                                   total_int * 100, 1),
               n_theoretical=len({round(io["mz"], 3) for io in ions}))
    return dict(mol=mol, frags=frags, ions=ions, peaks_table=rows, coverage=cov)


_SPEC_ALIASES = {"m/z": "mz", "m_z": "mz", "mass_to_charge": "mz",
                 "relative_intensity": "rel_intensity", "rel_int": "rel_intensity",
                 "relint": "rel_intensity", "abundance": "intensity",
                 "precursor": "precursor_mz", "precursormz": "precursor_mz",
                 "precursor_m/z": "precursor_mz", "parent_mz": "precursor_mz"}
_ANN_ALIASES = {"structure": "smiles", "canonical_smiles": "smiles",
                "compound": "name", "compound_name": "name", "identifier": "id",
                "hmdb_id": "id", "ion": "adduct", "ion_type": "adduct"}


_SCORE_DP = 4


def submission_table(cands, feature_id=None):
    """The candidate ranking in the form the blind benchmark is submitted in: one row"""
    rows = []
    for c in cands:
        rows.append(dict(
            feature_id=feature_id, name=c.get("name", ""), id=c.get("id", ""),
            adduct=c.get("adduct", ""), via=c.get("via", ""),
            score=round(float(c.get("score", 0.0)), _SCORE_DP),
            n_theoretical=int(c["coverage"]["n_theoretical"]),
            n_matched=int(c["coverage"]["n_matched"]),


            n_explained=int(c.get("score_parts", {}).get("n_explained", 0)),
            n_bonds_broken=c.get("n_bonds_broken"),


            _ions=frozenset(round(io["mz"], 4) for io in c.get("ions", [])),
        ))
    scores = [r["score"] for r in rows]
    all_zero = (max(scores) == 0) if scores else True
    icount = collections.Counter(r["_ions"] for r in rows)


    for r in rows:
        r["rank_best"] = 1 + sum(1 for s in scores if s > r["score"])
        r["rank_worst"] = sum(1 for s in scores if s >= r["score"])
        r["n_tied"] = sum(1 for s in scores if s == r["score"])
        r["identical_ions"] = icount[r["_ions"]] > 1
        r["all_zero"] = all_zero


    def key(r):
        nb = r["n_bonds_broken"]
        return (-r["score"], r["n_theoretical"], -r["n_explained"],
                nb if nb is not None else float("inf"))
    keys = [key(r) for r in rows]
    for r, k in zip(rows, keys):
        r["rank_tiebreak"] = sum(1 for k2 in keys if k2 <= k)

    out = pd.DataFrame(rows).drop(columns=["_ions"])
    return out.sort_values(["rank_tiebreak", "rank_worst"]).reset_index(drop=True)


def bonds_broken(c, method=DEFAULT_METHOD, max_breaks=MAX_BREAKS):
    """How many bonds had to be cut to produce the fragments this candidate matched."""
    fmap = fragment_map(c["mol"], method, max_breaks)
    num = den = 0.0
    per = {}
    for r in c["peaks_table"]:
        if not r["matched"] or not r.get("frag_smiles"):
            continue
        ent = fmap.get(r["frag_smiles"])
        if ent is None:
            continue
        n = len(ent[1])
        per[r["exp_mz"]] = n
        num += n * r["rel_int"]
        den += r["rel_int"]
    return (round(num / den, 3) if den else None), per


def demo_files(feat, data_dir=DATA):
    """(spectrum_file, annotations_file) for one of the two example features"""
    return (os.path.join(data_dir, f"{feat}_ms2_spectrum.csv"),
            os.path.join(data_dir, f"{feat}_ipa_annotations.csv"))


def _read_csv(path, what):
    """pd.read_csv on one of the two input files. Accepts a full path, a relative one"""
    full = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"{what} file not found: {full}\n"
            f"(for this project's example features: bf.demo_files('FT6080'))")
    df = pd.read_csv(full)
    aliases = _SPEC_ALIASES if what == "spectrum" else _ANN_ALIASES
    df.columns = [aliases.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    return df


def _one_feature(df, feature_id, what, notes):
    """These files carry a `feature_id` column, i.e. they COULD hold several features."""
    if "feature_id" not in df.columns:
        return df
    ids = list(dict.fromkeys(df["feature_id"].astype(str)))
    if feature_id is not None:
        sub = df[df["feature_id"].astype(str) == str(feature_id)]
        if not len(sub):
            raise ValueError(f"feature_id '{feature_id}' not in the {what} file "
                             f"(it has: {', '.join(ids)})")
        return sub
    if len(ids) > 1:
        raise ValueError(f"the {what} file holds {len(ids)} features "
                         f"({', '.join(ids[:5])}...). Pass feature_id= to pick one.")
    return df


def _load_all(spectrum_file, annotations_file, min_rel_int=None,
              drop_above_precursor=None, feature_id=None):
    """Read + normalise + filter BOTH input files in one pass, and report what it did."""


    if min_rel_int is None:
        min_rel_int = MIN_REL_INT
    if drop_above_precursor is None:
        drop_above_precursor = DROP_ABOVE_PRECURSOR
    notes = []
    ann = _one_feature(_read_csv(annotations_file, "annotations"),
                       feature_id, "annotations", notes)
    ms2 = _one_feature(_read_csv(spectrum_file, "spectrum"),
                       feature_id, "spectrum", notes).sort_values("mz")


    if "mz" not in ms2.columns:
        raise ValueError(f"the spectrum file has no m/z column (found: "
                         f"{', '.join(ms2.columns)})")
    if "smiles" not in ann.columns:
        raise ValueError(f"the annotation file has no SMILES column (found: "
                         f"{', '.join(ann.columns)})")
    if "rel_intensity" not in ms2.columns:
        if "intensity" not in ms2.columns:
            raise ValueError("the spectrum file has neither rel_intensity nor intensity")
        top = float(ms2["intensity"].max()) or 1.0
        ms2 = ms2.assign(rel_intensity=100 * ms2["intensity"] / top)
        notes.append("no rel_intensity column -> computed as 100 * intensity / max")
    if "name" not in ann.columns:
        ann = ann.assign(name=ann["id"].astype(str) if "id" in ann.columns
                         else [f"candidate {i+1}" for i in range(len(ann))])
        notes.append("no name column -> candidates named from id / position")
    if "id" not in ann.columns:
        ann = ann.assign(id="")
        notes.append("no id column -> left blank")

    prec = float(ms2["precursor_mz"].iloc[0]) if "precursor_mz" in ms2.columns else None
    adduct = str(ann["adduct"].iloc[0]) if "adduct" in ann.columns else None
    if prec is None:
        notes.append("no precursor_mz column -> the >= precursor filter is SKIPPED "
                     "and the neutral losses have no precursor to work from")
    if adduct is None:
        notes.append("no adduct column -> the precursor is assumed to be [M+H]+")


    def _status(mz, ri):
        if drop_above_precursor and prec is not None and mz >= prec:
            return "above_precursor"
        if ri < min_rel_int:
            return "below_noise"
        return "kept"

    n_raw = len(ms2)
    peaks_raw = [dict(mz=float(mz), rel_int=float(ri), status=_status(mz, ri))
                 for mz, ri in ms2[["mz", "rel_intensity"]].itertuples(index=False,
                                                                       name=None)]
    above = sum(1 for p in peaks_raw if p["status"] == "above_precursor")
    noise = sum(1 for p in peaks_raw if p["status"] == "below_noise")
    peaks = [(p["mz"], p["rel_int"]) for p in peaks_raw if p["status"] == "kept"]
    return dict(ann=ann, peaks=peaks, peaks_raw=peaks_raw, precursor_mz=prec,
                adduct=adduct, notes=notes, min_rel_int=min_rel_int,
                drop_above_precursor=drop_above_precursor,
                n_raw=n_raw, n_above=above, n_noise=noise)


def load(spectrum_file, annotations_file, min_rel_int=None, drop_above_precursor=None,
         feature_id=None):
    """The two input files in, (annotation_df, peaks) out - peaks being a list of"""
    d = _load_all(spectrum_file, annotations_file, min_rel_int,
                  drop_above_precursor, feature_id)
    return d["ann"], d["peaks"]


def read_precursor(spectrum_file, feature_id=None):
    """Precursor (parent-ion) m/z from the spectrum file - the cut-off used by"""
    ms2 = _one_feature(_read_csv(spectrum_file, "spectrum"), feature_id, "spectrum", [])
    return float(ms2["precursor_mz"].iloc[0]) if "precursor_mz" in ms2.columns else None


def analyze_candidates(ann, peaks, precursor_mz=None, method=DEFAULT_METHOD,
                       max_breaks=MAX_BREAKS, adduct=None, score=True,
                       include_generic=False, bonds=True):
    """The candidate loop, on data already in memory: analyze() every row of the"""
    forced = adduct
    has_col = "adduct" in ann.columns
    if adduct is None and has_col:
        adduct = str(ann["adduct"].iloc[0])
    cands = []
    for _, r in ann.iterrows():


        a = (forced if forced is not None else
             str(r["adduct"]) if (has_col and pd.notna(r.get("adduct"))) else adduct)
        res = analyze(peaks, str(r["smiles"]), method, max_breaks,
                      precursor_mz=precursor_mz, adduct=a)
        res["name"], res["id"], res["adduct"] = r["name"], r["id"], a
        res["via"] = res["frags"][0]["via"] if res["frags"] else "none"
        if bonds:
            res["n_bonds_broken"], _ = bonds_broken(res, method, max_breaks)
        if score:
            parts = candidate_score(res, a, include_generic)
            res["score"], res["score_parts"] = parts["score"], parts
            res["score_generic"] = include_generic
        cands.append(res)
    cands.sort(key=lambda c: (c.get("score", c["coverage"]["pct_intensity"]),
                              c["coverage"]["pct_intensity"]), reverse=True)
    return cands


def analyze_all(spectrum_file, annotations_file, method=DEFAULT_METHOD,
                max_breaks=MAX_BREAKS, adduct=None, score=True, include_generic=False,
                min_rel_int=None, drop_above_precursor=None, feature_id=None):
    """Read the two input files and run analyze_candidates() over EVERY candidate."""
    d = _load_all(spectrum_file, annotations_file, min_rel_int,
                  drop_above_precursor, feature_id)


    return analyze_candidates(d["ann"], d["peaks"], d["precursor_mz"], method,
                              max_breaks, adduct, score, include_generic)


def _find(cands, name):
    for c in cands:
        if name in (c["name"], c["id"]):
            return c
    raise KeyError(f"candidate not found: {name}")


def candidate(R, name=None):
    """One candidate out of a result bundle, by name; default = the best-scoring one."""
    return _find(R["candidates"], name or R["best"])


_ISOTOPE = {"H": 1.00782503207, "C": 12.0, "N": 14.0030740048,
            "O": 15.9949146196, "P": 30.97376163, "S": 31.97207100,
            "Na": 22.9897692809, "K": 38.96370668, "Cl": 34.96885268}


def _parse_formula(formula):
    """'C5H14NO4P' -> {'C':5,'H':14,'N':1,'O':4,'P':1}. A bare element means 1."""
    counts = {}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem:
            counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts


def formula_mass(formula):
    """Monoisotopic mass of a neutral formula string (so we never hand-type masses)."""
    return sum(_ISOTOPE[e] * n for e, n in _parse_formula(formula).items())


_UNIVERSAL = [
    ("H2O",  "water"),
    ("NH3",  "ammonia"),
    ("CO",   "carbon monoxide"),
    ("CO2",  "carbon dioxide"),
    ("CH2O", "formaldehyde"),
    ("C2H4", "ethylene"),
    ("C2H2", "acetylene"),
    ("H2",   "H2 (one degree of unsaturation)"),
    ("CH2",  "CH2 (homolog spacing)"),
]


_LOSS_RULES_SMARTS = [
    ("[NX4+](C)(C)(C)CCOP", [
        ("C3H9N",     "trimethylamine (choline head)"),
        ("HPO3",      "metaphosphate (choline head)"),
        ("H3PO4",     "phosphoric acid (choline head)"),
        ("C5H14NO4P", "phosphocholine head group"),
    ]),
    ("[NX3][CH2][CH2]O[PX4]", [
        ("C2H8NO4P", "phosphoethanolamine head group"),
        ("C2H7NO",   "ethanolamine"),
    ]),
    ("[PX4](=O)([OX2H1,OX1-])[OX2H1,OX1-]", [
        ("HPO3",  "metaphosphate"),
        ("H3PO4", "phosphoric acid"),
    ]),
    ("[CX3](=O)[OX2H1]", [
        ("CO2", "CO2 (decarboxylation)"),
    ]),
    ("[OX2,OX1-]S(=O)(=O)[OX2,OX1-]", [
        ("SO3", "sulfur trioxide"),
    ]),


    ("[OX2;!$([OX2]C=O)][CX4H1]1[OX2][CX4H1]([CH2][OX2])[CX4H1]([OX2H1])"
     "[CX4H1]([OX2H1])[CX4H1]1[OX2H1]", [


        ("C6H10O5", "hexose residue (glycosidic cleavage)"),
        ("C6H12O6", "free hexose (glycoside + H2O)"),
    ]),
    ("[OX2;!$([OX2]C=O)][CX4H1]1[OX2][CX4H1]([CH3])[CX4H1]([OX2H1])"
     "[CX4H1]([OX2H1])[CX4H1]1[OX2H1]", [
        ("C6H10O4", "deoxyhexose residue"),
    ]),
    ("[OX2;!$([OX2]C=O)][CX4H1]1[OX2][CH2][CX4H1]([OX2H1])[CX4H1]([OX2H1])"
     "[CX4H1]1[OX2H1]", [
        ("C5H8O4", "pentose residue"),
    ]),
    ("[OX2;!$([OX2]C=O)][CX4H1]1[OX2][CX4H1]([CH2][OX2H1])[CX4H1]([OX2H1])"
     "[CX4H1]1[OX2H1]", [
        ("C5H8O4", "pentose residue"),
    ]),
    ("[OX2][CX4H1]1[OX2][CX4H1]([CX3](=O)[OX2H1,OX1-])[CX4H1]([OX2H1])"
     "[CX4H1]([OX2H1])[CX4H1]1[OX2H1]", [
        ("C6H8O6", "glucuronic acid residue"),
    ]),
    ("[#6][OX2][CH3]", [
        ("CH4O", "methanol (methoxy)"),
    ]),
    ("[#6][OX2][CX3](=O)[CH3]", [
        ("C2H2O",  "ketene (O-acetyl)"),
        ("C2H4O2", "acetic acid (O-acetyl)"),
    ]),
    ("[NX3][CX3](=O)[CH3]", [
        ("C2H2O", "ketene (N-acetyl)"),
    ]),
    ("[CX3](=[OX1])[NX3H2]", [
        ("NH3", "ammonia (primary amide)"),
    ]),
    ("[NX3][CX3](=[NX2])[NX3]", [
        ("NH3",   "ammonia (guanidine)"),
        ("CH2N2", "carbodiimide (guanidine)"),
    ]),
    ("[NX3][CX3](=[OX1,SX1])[NX3]", [
        ("NH3",  "ammonia (urea)"),
        ("CHNO", "isocyanic acid (urea)"),
    ]),
    ("[$([NX3](=O)=O),$([NX3+](=O)[O-])]", [
        ("HNO2", "nitrous acid (nitro)"),
    ]),
    ("[Cl][#6]", [
        ("HCl", "hydrogen chloride"),
    ]),
    ("[#6][SX2][#6,#1]", [
        ("H2S", "hydrogen sulfide"),
    ]),
    ("[NX4+]([CH3])([CH3])[CH3]", [
        ("C3H9N", "trimethylamine (quaternary N)"),
    ]),
    ("[#6][SX4](=O)(=O)[OX2H1,OX1-]", [
        ("SO3", "sulfur trioxide (sulfonate)"),
    ]),
]
_LOSS_RULES = [(Chem.MolFromSmarts(s), losses) for s, losses in _LOSS_RULES_SMARTS]

_ACYL_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
_ACID_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")


_MIN_ACYL_CARBONS = 8


def _is_fatty_acid(acid_mol):
    """Is this RCOOH really a FATTY acid, i.e. does the '16:1' notation mean anything?"""
    if acid_mol.GetRingInfo().NumRings():
        return False
    c = _parse_formula(rdMolDescriptors.CalcMolFormula(acid_mol))
    if set(c) - {"C", "H", "O"}:
        return False
    return c.get("C", 0) >= _MIN_ACYL_CARBONS


def _acid_label(acid_mol):
    """'16:0' style label (carbons : C=C in the chain) from an RCOOH fragment."""
    c = _parse_formula(rdMolDescriptors.CalcMolFormula(acid_mol))
    nc, nh, nn = c.get("C", 0), c.get("H", 0), c.get("N", 0)
    dou = (2 * nc + 2 + nn - nh) / 2
    return f"{nc}:{max(int(dou) - 1, 0)}"


def acyl_losses(mol):
    """LIPID-ONLY. The fatty-acyl chains this molecule can lose. For each ester chain"""
    out = {}
    for m_ in mol.GetSubstructMatches(_ACYL_SMARTS):
        cC, _o, oEst, _g = m_
        bond = mol.GetBondBetweenAtoms(cC, oEst).GetIdx()
        frag = Chem.FragmentOnBonds(mol, [bond], addDummies=True)
        for piece in Chem.MolToSmiles(frag).split("."):
            if "*" not in piece:
                continue
            acid = Chem.MolFromSmiles(re.sub(r"\[\d*\*\]|\*", "O", piece))
            if acid is None or not acid.HasSubstructMatch(_ACID_SMARTS):
                continue
            acid_mass = ExactMolWt(acid)
            formula = rdMolDescriptors.CalcMolFormula(acid)
            if _is_fatty_acid(acid):
                key = _acid_label(acid)
                name = f"fatty acid {key} (RCOOH)"
            else:
                key = formula
                name = f"acid {formula} (ester cleavage)"
            out[key] = dict(mass=acid_mass, formula=formula, name=name, source="acyl")
    losses = list(out.values())
    for a in list(out.values()):
        ketene = (a["name"].replace("(RCOOH)", "as ketene (-H2O)")
                  if "(RCOOH)" in a["name"]
                  else a["name"].replace("(ester cleavage)", "as ketene (-H2O)"))
        losses.append(dict(mass=a["mass"] - formula_mass("H2O"),
                           formula=a["formula"], source="acyl", name=ketene))
    return losses


def candidate_losses(mol, adduct=None):
    """Library 2 as an ALGORITHM: all neutral losses this precursor structure can"""
    losses = [dict(mass=formula_mass(f), formula=f, name=nm, source="universal")
              for f, nm in _UNIVERSAL]
    heads = []
    for query, rules in _LOSS_RULES:
        if query is not None and mol.HasSubstructMatch(query):
            heads += [dict(mass=formula_mass(f), formula=f, name=nm, source="headgroup")
                      for f, nm in rules]
    acyls = acyl_losses(mol)
    losses += heads + acyls
    if adduct and "NH4" in adduct:


        nh3 = formula_mass("NH3")
        for a in heads + acyls:
            if a["formula"] == "NH3":
                continue
            src = "adduct+acyl" if a["source"] == "acyl" else "adduct+headgroup"
            losses.append(dict(mass=nh3 + a["mass"], formula=f'NH3+{a["formula"]}',
                               name=f'NH3 + {a["name"]}', source=src))
    seen, uniq = set(), []
    for L in losses:
        key = (round(L["mass"], 4), L["name"])
        if key not in seen:
            seen.add(key)
            uniq.append(L)
    return uniq


_SRC_WEIGHT = {"acyl": 1.0, "adduct+acyl": 1.0, "headgroup": 0.7,
               "adduct": 0.5, "universal": 0.4}


def match_loss(delta, losses):
    """Explain a mass difference: EVERY loss whose mass equals |delta| within the"""
    d = abs(delta)
    tol = max(TOL_MIN, d * TOL_PPM * 1e-6)
    hits = [dict(L, ppm=round((d - L["mass"]) / d * 1e6, 1) if d else 0.0)
            for L in losses if abs(d - L["mass"]) <= tol]
    sigma = TOL_PPM / 2 or 1.0
    for h in hits:
        h["_s"] = _SRC_WEIGHT.get(h["source"], 0.3) * math.exp(-0.5 * (h["ppm"] / sigma) ** 2)
    z = sum(h["_s"] for h in hits) or 1.0
    for h in hits:
        h["P"] = round(h["_s"] / z, 3)
        del h["_s"]
    hits.sort(key=lambda h: -h["P"])
    return hits


def _theoretical_precursor(mol, adduct):
    """Theoretical precursor m/z = neutral mass + adduct (removes the ~0.016 Da"""
    neutral = ExactMolWt(mol)
    norm = re.sub(r"[^A-Za-z0-9]", "", adduct or "")
    for name, a in all_adducts().items():
        if re.sub(r"[^A-Za-z0-9]", "", name) == norm:
            return adduct_mz(neutral, a)
    return neutral + PROTON


def _fmt_all(hits):
    """Compact 'all explanations': 'name(P); name(P); ...' sorted best first."""
    return "; ".join(f'{h["name"]} ({h["P"]})' for h in hits)


_SPACING_FORMULAS = {"CH2", "C2H2", "C2H4", "H2"}
_CAT_RANK = {"acyl bridge": 0, "head-group": 1, "adduct swap": 1.5, "small loss": 2,
             "series spacing": 3, "(no loss)": 4}


def _loss_category(source, formula, matched):
    if not matched:
        return "(no loss)"
    if source in ("acyl", "adduct+acyl"):
        return "acyl bridge"
    if source in ("headgroup", "adduct+headgroup"):
        return "head-group"
    if formula in _SPACING_FORMULAS:
        return "series spacing"
    return "small loss"


def _adduct_lookup():
    """{adduct name -> its dict} for back-calculating neutral cores. Every adduct in"""
    return dict(all_adducts())


def _neutral_core(mz, adduct_name, lut):
    """A peak's neutral mass, obtained by stripping its adduct; None if the adduct is"""
    a = lut.get(adduct_name)
    return None if a is None else (mz - a["mass"]) * abs(a["charge"]) / a["mult"]


def _adduct_swap_pair(mz_hi, a_hi, mz_lo, a_lo, tol, lut):
    """Label if (hi, lo) are one neutral species under two different adducts, else None."""
    if not (a_hi and a_lo) or a_hi == a_lo:
        return None
    ch, cl = _neutral_core(mz_hi, a_hi, lut), _neutral_core(mz_lo, a_lo, lut)
    if ch is None or cl is None or abs(ch - cl) > tol:
        return None
    return f"{a_lo} / {a_hi} (same neutral {ch:.4f})"


def precursor_losses(c, adduct, matched_only=False):
    """Task 2 step 1: neutral loss = theoretical precursor - each peak, annotated."""
    mol = c["mol"]
    prec = _theoretical_precursor(mol, adduct)
    losses = candidate_losses(mol, adduct=adduct)
    M = ExactMolWt(mol)
    prec_norm = re.sub(r"[^A-Za-z0-9]", "", adduct or "")


    other_mz = {a["name"]: adduct_mz(M, a)
                for a in get_fragment_adducts(adduct_mode(adduct))}
    rows = []
    for r in c["peaks_table"]:
        delta = prec - r["exp_mz"]
        tol = max(TOL_MIN, r["exp_mz"] * TOL_PPM * 1e-6)
        swap = next((nm for nm, amz in other_mz.items()
                     if re.sub(r"[^A-Za-z0-9]", "", nm) != prec_norm
                     and abs(r["exp_mz"] - amz) <= tol), None)
        if swap and delta > 1.5 and not r["special"]:
            rows.append(dict(exp_mz=r["exp_mz"], rel_int=r["rel_int"], delta=round(delta, 4),
                             matched=True, category="adduct swap",
                             loss=f"{swap} of intact molecule (adduct swap, not a loss)",
                             formula="", source="adduct-swap", ppm=None, P=None,
                             n_expl=0, all_expl=""))
            continue
        hits = match_loss(delta, losses) if delta > 1.5 else []
        best = hits[0] if hits else None
        rows.append(dict(exp_mz=r["exp_mz"], rel_int=r["rel_int"], delta=round(delta, 4),
                         matched=bool(hits),
                         category=_loss_category(best["source"] if best else "",
                                                 best["formula"] if best else "", bool(hits)),
                         loss=best["name"] if best else "",
                         formula=best["formula"] if best else "",
                         source=best["source"] if best else "",
                         ppm=best["ppm"] if best else None,
                         P=best["P"] if best else None,
                         n_expl=len(hits), all_expl=_fmt_all(hits)))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("rel_int", ascending=False).reset_index(drop=True)
    if matched_only and len(df):
        df = df[df["matched"]].reset_index(drop=True)
    return df


def pairwise_losses(c, adduct, min_gap=1.5, matched_only=False):
    """Task 2 step 2: neutral loss = peak(high) - peak(low), annotated. Guards drop"""
    mol = c["mol"]
    losses = candidate_losses(mol, adduct=None)
    lut = _adduct_lookup()
    pk = [(r["exp_mz"], r["rel_int"], r.get("best_adduct", "")) for r in c["peaks_table"]]
    c13 = 1.003355
    rows = []
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            a, b = pk[i], pk[j]
            (hi, ih, a_hi), (lo, il, a_lo) = (a, b) if a[0] >= b[0] else (b, a)
            delta = hi - lo
            if delta < min_gap:
                continue
            if abs(delta - c13) <= max(TOL_MIN, delta * TOL_PPM * 1e-6):
                continue
            swap = _adduct_swap_pair(hi, a_hi, lo, a_lo,
                                     max(TOL_MIN, hi * TOL_PPM * 1e-6), lut)
            if swap:
                rows.append(dict(mz_hi=round(hi, 4), mz_lo=round(lo, 4),
                                 int_hi=round(ih, 3), int_lo=round(il, 3),
                                 delta=round(delta, 4), matched=True,
                                 category="adduct swap", loss=swap, formula="",
                                 source="adduct-swap", ppm=None, P=None,
                                 n_expl=0, all_expl=""))
                continue
            hits = match_loss(delta, losses)
            best = hits[0] if hits else None
            rows.append(dict(mz_hi=round(hi, 4), mz_lo=round(lo, 4),
                             int_hi=round(ih, 3), int_lo=round(il, 3),
                             delta=round(delta, 4), matched=bool(hits),
                             category=_loss_category(best["source"] if best else "",
                                                     best["formula"] if best else "", bool(hits)),
                             loss=best["name"] if best else "",
                             formula=best["formula"] if best else "",
                             source=best["source"] if best else "",
                             ppm=best["ppm"] if best else None,
                             P=best["P"] if best else None,
                             n_expl=len(hits), all_expl=_fmt_all(hits)))
    df = pd.DataFrame(rows, columns=_PAIRWISE_COLS)
    if len(df):
        df["_r"] = df["category"].map(_CAT_RANK).fillna(9)
        df = df.sort_values(["_r", "mz_hi"], ascending=[True, False]).drop(columns="_r").reset_index(drop=True)
    if matched_only and len(df):
        df = df[df["matched"]].reset_index(drop=True)
    return df


_DIAGNOSTIC_CATS = {"acyl bridge", "head-group"}


_PAIRWISE_COLS = ["mz_hi", "mz_lo", "int_hi", "int_lo", "delta", "matched", "category",
                  "loss", "formula", "source", "ppm", "P", "n_expl", "all_expl"]


def _better(store, key, cat, P, txt):
    """Keep, per peak, the MOST DIAGNOSTIC explanation (lowest category rank),"""
    rank = _CAT_RANK.get(cat, 9)
    cur = store.get(key)
    if cur is None or (rank, -(P or 0)) < (cur[0], -cur[1]):
        store[key] = (rank, P or 0, cat, txt)


def _rescue_tier(cat):
    if cat in _DIAGNOSTIC_CATS:
        return "diagnostic"
    if cat:
        return "generic"
    return "unexplained"


def _rescue(c, adduct):
    """loss_rescue() memoised on the candidate. The neutral-loss pass is the expensive"""
    hit = c.get("_rescue_cache")
    if hit is None or hit[0] != adduct:
        c["_rescue_cache"] = (adduct, loss_rescue(c, adduct))
    return c["_rescue_cache"][1]


def loss_rescue(c, adduct):
    """B1 - tie back to Task 1, split by how MEANINGFUL the rescue is. Of the peaks"""
    unexp = [(r["exp_mz"], r["rel_int"]) for r in c["peaks_table"]
             if (not r["matched"]) and r["special"] == ""]
    unexp_mz = {round(mz, 4) for mz, _ in unexp}
    via = {}
    pl = precursor_losses(c, adduct)
    for _, r in pl[pl["matched"] & (pl["source"] != "adduct-swap")].iterrows():
        mz = round(r["exp_mz"], 4)
        if mz in unexp_mz:
            _better(via, mz, r["category"], r["P"], f'precursor - {r["delta"]:.3f} ({r["loss"]})')
    pw = pairwise_losses(c, adduct)
    for _, r in pw[pw["matched"] & (pw["source"] != "adduct-swap")].iterrows():
        txt = f'{r["mz_hi"]:.3f}-{r["mz_lo"]:.3f}={r["delta"]:.3f} ({r["loss"]})'
        for mz in (round(r["mz_hi"], 4), round(r["mz_lo"], 4)):
            if mz in unexp_mz:
                _better(via, mz, r["category"], r["P"], txt)
    rows = []
    for mz, ri in sorted(unexp, key=lambda x: -x[1]):
        v = via.get(round(mz, 4))
        cat = v[2] if v else ""
        rows.append(dict(exp_mz=mz, rel_int=ri, tier=_rescue_tier(cat),
                         category=cat, via=v[3] if v else ""))
    df = pd.DataFrame(rows)
    total_int = sum(r["rel_int"] for r in c["peaks_table"]) or 1

    def _tier(t):
        sub = df[df["tier"] == t] if len(df) else df
        return len(sub), round(100 * (sub["rel_int"].sum() if len(sub) else 0.0) / total_int, 1)

    dP, dI = _tier("diagnostic")
    gP, gI = _tier("generic")
    uP, uI = _tier("unexplained")
    task1_int = round(100 * sum(r["rel_int"] for r in c["peaks_table"]
                                if r["matched"] or r["special"]) / total_int, 1)
    summary = dict(task1_explained_pct=task1_int,
                   rescued_diagnostic_pct=dI, rescued_diagnostic_peaks=dP,
                   rescued_generic_pct=gI, rescued_generic_peaks=gP,
                   still_unexplained_pct=uI, still_unexplained_peaks=uP,
                   n_unexplained=len(unexp))
    return summary, df


def candidate_score(c, adduct, include_generic=False):
    """Score ONE candidate: the share of the total measured MS2 intensity this"""
    _, resc = _rescue(c, adduct)
    peaks = c["peaks_table"]
    total = sum(r["rel_int"] for r in peaks) or 1.0
    t1_int = sum(r["rel_int"] for r in peaks if r["matched"] or r["special"])
    t1_n = sum(1 for r in peaks if r["matched"] or r["special"])

    def _tier(t):
        sub = resc[resc["tier"] == t] if len(resc) else resc
        return (len(sub), float(sub["rel_int"].sum())) if len(sub) else (0, 0.0)

    d_n, d_int = _tier("diagnostic")
    g_n, g_int = _tier("generic")
    expl_int = t1_int + d_int + (g_int if include_generic else 0.0)
    expl_n = t1_n + d_n + (g_n if include_generic else 0)
    return dict(score=round(100 * expl_int / total, 1),
                task1_pct=round(100 * t1_int / total, 1),
                task2_diagnostic_pct=round(100 * d_int / total, 1),
                task2_generic_pct=round(100 * g_int / total, 1),
                unexplained_pct=round(100 * (total - expl_int) / total, 1),
                n_explained=expl_n, n_peaks=len(peaks))


def score_table(cands, adduct=None, include_generic=False):
    """The scoreboard: one row per candidate annotation, best score first."""
    rows = []
    for c in cands:
        parts = c.get("score_parts")
        if parts is None or c.get("score_generic") != include_generic:
            parts = candidate_score(c, adduct or c.get("adduct"), include_generic)
        rows.append(dict(name=c["name"] or c["id"], id=c["id"], **parts))
    return (pd.DataFrame(rows).sort_values("score", ascending=False)
            .reset_index(drop=True))


def style_scores(cands, adduct=None, include_generic=False):
    """score_table() as a pandas Styler: the score in bold as the headline number,"""
    df = score_table(cands, adduct, include_generic)
    faded = ["unexplained_pct"] if include_generic else ["task2_generic_pct",
                                                         "unexplained_pct"]
    return (df.style
            .format({c: "{:.1f}" for c in ["score", "task1_pct", "task2_diagnostic_pct",
                                           "task2_generic_pct", "unexplained_pct"]})
            .set_properties(subset=["score"], **{"font-weight": "700"})
            .set_properties(subset=faded, opacity="0.45"))


def explained_table(c, adduct):
    """ONE row per measured peak, saying what - if anything - explains it, merging the"""
    _, resc = _rescue(c, adduct)
    via = {round(r["exp_mz"], 4): (r["tier"], r["category"], r["via"])
           for _, r in resc.iterrows()} if len(resc) else {}
    rows = []
    for r in c["peaks_table"]:
        kind, detail = r["kind"], ""
        if r["matched"]:
            detail = (f'{r["name"] or r["formula"]} [{r["best_adduct"]}] '
                      f'{r["theo_mz"]:.4f} ({r["ppm"]:+.1f} ppm)')
        elif r["special"]:
            detail = r["special"]
        else:
            tier, cat, txt = via.get(r["exp_mz"], ("unexplained", "", ""))
            if tier != "unexplained":
                kind, detail = tier, f"{cat}: {txt}"
        rows.append(dict(exp_mz=r["exp_mz"], rel_int=r["rel_int"], kind=kind,
                         counted=kind in ("fragment", "isotope", "precursor",
                                          "diagnostic"),
                         detail=detail, frag_smiles=r["frag_smiles"],
                         name=r["name"], formula=r["formula"],
                         best_adduct=r["best_adduct"], theo_mz=r["theo_mz"],
                         ppm=r["ppm"], n_hits=r["n_hits"]))
    return (pd.DataFrame(rows).sort_values("rel_int", ascending=False)
            .reset_index(drop=True))


def explain_spectrum(spectrum_file, annotations_file, method="FragmentOnBonds",
                     max_breaks=MAX_BREAKS, ppm=None, min_da=None, min_rel_int=None,
                     drop_above_precursor=None, feature_id=None, adduct=None,
                     include_generic=False, verbose=True):
    """Score every candidate annotation in `annotations_file` against the MS2 spectrum"""
    if ppm is not None or min_da is not None:
        set_tolerance(TOL_PPM if ppm is None else ppm, min_da)
    d = _load_all(spectrum_file, annotations_file, min_rel_int,
                  drop_above_precursor, feature_id)
    adduct = adduct or d["adduct"]
    cands = analyze_candidates(d["ann"], d["peaks"], d["precursor_mz"], method,
                               max_breaks, adduct, True, include_generic)
    scores = score_table(cands, adduct, include_generic)
    explained = {c["name"]: explained_table(c, adduct) for c in cands}
    kept = len(d["peaks"])
    if verbose:
        print("spectrum   :", os.path.basename(str(spectrum_file)))
        print("annotations:", os.path.basename(str(annotations_file)),
              f"({len(d['ann'])} candidates)")
        print("settings   :", method,
              f"| N = {max_breaks} | ppm = {TOL_PPM:g} | min_da = {TOL_MIN:g}")
        print("peaks      :", f"{d['n_raw']} raw -> {kept} kept"
              f" ({d['n_above']} at/above the precursor,"
              f" {d['n_noise']} below the noise floor)")
        print("precursor  :", f"{d['precursor_mz']:.4f}" if d["precursor_mz"] else "-",
              f"[{adduct}]" if adduct else "")
        for n in d["notes"]:
            print("  note     :", n)
        if len(scores):
            print("best score :", f"{scores['score'].iloc[0]:.1f}%",
                  "-", scores["name"].iloc[0])
    return dict(scores=scores, candidates=cands, explained=explained,
                best=scores["name"].iloc[0] if len(scores) else None,
                ann=d["ann"], peaks=d["peaks"], peaks_raw=d["peaks_raw"],
                precursor_mz=d["precursor_mz"],
                adduct=adduct, spectrum_file=spectrum_file,
                annotations_file=annotations_file, method=method,
                max_breaks=max_breaks, ppm=TOL_PPM, min_da=TOL_MIN,
                min_rel_int=d["min_rel_int"],
                drop_above_precursor=d["drop_above_precursor"],
                input_notes=d["notes"],
                dropped=dict(raw=d["n_raw"], above_precursor=d["n_above"],
                             below_noise=d["n_noise"], kept=kept))


_SVG_CACHE = {}


def _svg_mol(mol, w, h, keep_classes=False):
    """Draw a molecule. keep_classes leaves RDKit's class='bond-N atom-A atom-B' /"""
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    d.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return _minify_svg(d.GetDrawingText(), w, h, keep_classes)


def _minify_svg(s, w, h, keep_classes=False):
    """RDKit writes the same `fill:none;stroke:#000000;stroke-width:2.0px;...` onto"""
    s = s[s.index("<!-- END OF HEADER -->") + 22:]
    if not keep_classes:
        s = re.sub(r"\s*class='[^']*'", "", s)

    def _style(m):
        st, out = m.group(1), []
        col = re.search(r"stroke:(#[0-9A-Fa-f]{6})", st)
        if col and col.group(1) != "#000000":
            out.append("stroke:" + col.group(1))
        fil = re.search(r"fill:(#[0-9A-Fa-f]{6})", st)
        if fil:
            out.append("fill:" + fil.group(1))
        wid = re.search(r"stroke-width:([\d.]+)", st)
        if wid and abs(float(wid.group(1)) - 2) > 0.01:
            out.append("stroke-width:" + wid.group(1))
        return (" style='%s'" % ";".join(out)) if out else ""

    s = re.sub(r" style='([^']*)'", _style, s)
    s = re.sub(r"\s+", " ", s).strip()
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' "
            f"width='100%' height='100%'><style>path{{fill:none;stroke:#000;"
            f"stroke-width:2}}</style>{s}</svg>")


def _svg(smiles, w=210, h=150):
    """One structure as a MINIFIED inline SVG. RDKit writes the same"""
    key = (smiles, w, h)
    if key in _SVG_CACHE:
        return _SVG_CACHE[key]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    out = _svg_mol(mol, w, h)
    _SVG_CACHE[key] = out
    return out


def _formation_site(mol, row, fmap, lsites):
    """How to paint ONE peak onto the parent structure, or None when there is nothing"""
    kind = row["kind"]
    if kind in ("fragment", "isotope", "precursor"):
        hit = fmap.get(row["frag_smiles"])
        if not hit:
            return None
        atoms, cut = hit
        name = row["name"] or row["formula"] or "this fragment"
        return dict(k="f", a=atoms, b=_bonds_in(mol, atoms), x=cut,
                    c=f"<b style='color:#2f9e44'>green</b> = {name}, the fragment that "
                      f"matches this peak; <b style='color:#e0524a'>dashed red</b> = "
                      f"the bond{'s' if len(cut) > 1 else ''} BRICS cut to release it")
    detail = str(row["detail"])
    for label, (atoms, cut) in sorted(lsites.items(), key=lambda kv: -len(kv[0])):
        if f"fatty acid {label}" in detail or label in detail:
            what = (f"the {label} acyl chain" if ":" in label else f"the {label}")
            return dict(k="l", a=atoms, b=_bonds_in(mol, atoms), x=cut,
                        c=f"<b style='color:#e0524a'>red</b> = {what}, lost as a "
                          f"neutral; <b style='color:#1a56c4'>blue</b> = the part that "
                          f"stays behind and gives this peak")
    return None


def _app_payload(R, all_structures=False):
    """Everything the page needs, as one JSON-able dict. Structures are pooled in a"""
    svgs, idx = [], {}

    def svg_id(smi, w=210, h=150, make=True):
        if not smi:
            return -1
        key = (smi, w, h)
        if key not in idx:
            if not make:
                return -1
            idx[key] = len(svgs)
            svgs.append(_svg(smi, w, h))
        return idx[key]

    st_code = {"kept": 0, "below_noise": 1, "above_precursor": 2}
    peaks = [[round(p["mz"], 4), round(p["rel_int"], 3), st_code[p["status"]]]
             for p in R["peaks_raw"]]
    at = {round(p["mz"], 4): i for i, p in enumerate(R["peaks_raw"])}
    kept_mz = [mz for mz, _ in R["peaks"]]
    kind_code = {"fragment": 0, "isotope": 0, "precursor": 0,
                 "diagnostic": 1, "generic": 2, "": 3}

    def num(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)

    cands = []
    for c in R["candidates"]:


        fmap = fragment_map(c["mol"], R["method"], R["max_breaks"])
        sites, lsites = {}, loss_sites(c["mol"])
        exp = {}
        for _, r in R["explained"][c["name"]].iterrows():
            i = at.get(round(r["exp_mz"], 4))
            if i is None:
                continue
            exp[i] = dict(k=kind_code.get(r["kind"], 3), d=r["detail"],
                          c=bool(r["counted"]), n=r["name"], f=r["formula"],
                          a=r["best_adduct"], t=num(r["theo_mz"]), p=num(r["ppm"]),
                          s=svg_id(r["frag_smiles"]))
            site = _formation_site(c["mol"], r, fmap, lsites)
            if site:
                sites[str(i)] = site
        seen, theo = set(), []
        for io in c["ions"]:
            key = (round(io["mz"], 4), io["frag"], io["adduct"])
            if key in seen:
                continue
            seen.add(key)
            hit = any(match(mz, io["mz"]) for mz in kept_mz)
            theo.append([round(io["mz"], 4), io["formula"], io["adduct"], int(hit),
                         svg_id(io["frag"], make=all_structures)])
        parts = c.get("score_parts", {})
        svgs.append(_svg_mol(c["mol"], 470, 300, keep_classes=True))
        cands.append(dict(name=c["name"], id=c["id"],
                          smiles=Chem.MolToSmiles(c["mol"]),
                          score=c.get("score"), parts=parts,
                          mol=len(svgs) - 1, sites=sites,
                          exp={str(k): v for k, v in exp.items()}, theo=theo))
    meta = dict(spectrum=os.path.basename(str(R["spectrum_file"])),
                annotations=os.path.basename(str(R["annotations_file"])),
                method=R["method"], n=R["max_breaks"], ppm=R["ppm"],
                min_da=R["min_da"], min_rel_int=R.get("min_rel_int"),
                drop_above=bool(R.get("drop_above_precursor", True)),
                precursor=R["precursor_mz"], adduct=R["adduct"],
                dropped=R["dropped"], notes=R["input_notes"])
    return dict(meta=meta, peaks=peaks, svgs=svgs, cands=cands)


_APP_TEMPLATE = r"""
<style>
#AMROOT{--grey:#b9bfc7;--red:#e0524a;--green:#2f9e44;--blue:#1a56c4;--blue2:#93b8ee;
  --line:#dfe3e8;--ink:#22262b;--dim:#6b7280;
  background:#fff;color:var(--ink);font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
  display:grid;grid-template-columns:minmax(240px,330px) minmax(0,1fr);
  grid-template-areas:"q2 q1" "q3 q4";gap:10px;padding:12px;position:relative}
@media (max-width:900px){#AMROOT{grid-template-columns:minmax(0,1fr);
  grid-template-areas:"q1" "q4" "q3" "q2"}}
#AMROOT .q{border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:0}
#AMROOT .q1{grid-area:q1}#AMROOT .q2{grid-area:q2}
#AMROOT .q3{grid-area:q3}#AMROOT .q4{grid-area:q4}
#AMROOT h4{margin:0 0 8px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--dim);font-weight:600}
#AMROOT .kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:12px}
#AMROOT .kv b{font-weight:600;color:var(--dim)}
#AMROOT .mono{font-family:ui-monospace,Consolas,monospace}
#AMROOT .note{margin-top:8px;font-size:11.5px;color:#b45309;background:#fffbeb;
  border-radius:5px;padding:5px 7px}
#AMROOT .legend{display:flex;flex-wrap:wrap;gap:10px;font-size:11.5px;margin-bottom:4px;
  align-items:center}
#AMROOT .legend i{display:inline-block;width:9px;height:9px;border-radius:2px;
  margin-right:4px;vertical-align:-1px}
#AMROOT .tools{margin-left:auto;display:flex;gap:10px;align-items:center;color:var(--dim)}
#AMROOT .tools label{cursor:pointer}
#AMROOT button{font:inherit;font-size:11.5px;padding:2px 8px;border:1px solid var(--line);
  background:#f8fafc;border-radius:5px;cursor:pointer;color:var(--ink)}
#AMROOT button:hover{background:#eef2f7}
#AMROOT #AMplot{width:100%;height:auto;display:block;cursor:crosshair;user-select:none}
#AMROOT .row{display:grid;grid-template-columns:14.5em minmax(0,1fr) 3.4em;
  gap:4px 8px;align-items:baseline;padding:5px 7px;border-radius:6px;cursor:pointer;
  border:1px solid transparent}
#AMROOT .row:hover{background:#f3f6fa}
#AMROOT .row.on{background:#eaf1fd;border-color:#c3d7f7}
#AMROOT .row .nm{font-weight:600;font-size:12.5px;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
#AMROOT .row .sm{font-size:10.5px;color:var(--dim);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;font-family:ui-monospace,Consolas,monospace}
#AMROOT .sc{text-align:right;font-weight:700;font-size:13px}
#AMROOT .hd{font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:var(--dim);
  border-bottom:1px solid var(--line);padding-bottom:3px;margin-bottom:2px;cursor:default}
#AMROOT .hd:hover{background:none}
#AMROOT .bar{grid-column:1/-1;display:flex;height:5px;border-radius:3px;overflow:hidden;
  background:#eef1f5;margin-top:2px}
#AMROOT .bar i{display:block;height:100%}
#AMROOT .foot{font-size:11px;color:var(--dim);margin-top:7px;border-top:1px solid var(--line);
  padding-top:6px}
#AMROOT #AMlist{max-height:290px;overflow:auto}
#AMROOT #AMmol{height:274px;display:flex;align-items:center;justify-content:center}
#AMROOT .cap{font-size:11.5px;color:var(--dim);margin:-2px 0 4px;min-height:30px;
  line-height:1.35}
#AMROOT .cap b{color:var(--ink)}
#AMROOT #AMpop{position:absolute;z-index:20;display:none;width:290px;background:#fff;
  border:1px solid #c9d2dc;border-radius:9px;box-shadow:0 8px 24px rgba(15,23,42,.18);
  padding:10px 12px;font-size:12px}
#AMROOT #AMpop .x{position:absolute;top:5px;right:8px;cursor:pointer;color:var(--dim);
  font-size:15px;line-height:1}
#AMROOT #AMpop h5{margin:0 0 6px;font-size:13px}
#AMROOT #AMpop .tag{display:inline-block;font-size:10.5px;padding:1px 6px;border-radius:9px;
  color:#fff;margin-bottom:6px}
#AMROOT #AMpop .st{height:130px;margin-top:6px;border-top:1px solid var(--line);padding-top:6px}
#AMROOT #AMpop table{border-collapse:collapse;width:100%}
#AMROOT #AMpop td{padding:1px 0;vertical-align:top}
#AMROOT #AMpop td:first-child{color:var(--dim);padding-right:8px;white-space:nowrap}
</style>

<div id="AMROOT">
  <div class="q q2">
    <h4>Input &amp; settings</h4>
    <div class="kv" id="AMmeta"></div>
    <div id="AMnotes"></div>
  </div>

  <div class="q q1">
    <h4>Mirror spectrum <span style="text-transform:none;letter-spacing:0">
      &mdash; measured above, theoretical fragments below</span></h4>
    <div class="legend">
      <span><i style="background:var(--green)"></i>fragment match</span>
      <span><i style="background:var(--blue)"></i>neutral loss (counted)</span>
      <span><i style="background:var(--blue2)"></i>generic loss</span>
      <span><i style="background:var(--red)"></i>unexplained</span>
      <span><i style="background:var(--grey)"></i>removed</span>
    </div>
    <div class="legend" style="margin-bottom:6px">
      <span style="color:var(--dim)">show:</span>
      <label><input type="checkbox" id="AMred" checked> unexplained</label>
      <label><input type="checkbox" id="AMgrey" checked> removed</label>
      <label><input type="checkbox" id="AMonly"> matched theoretical only</label>
      <span class="tools"><button id="AMreset">reset zoom</button></span>
    </div>
    <svg id="AMplot" viewBox="0 0 1000 430" preserveAspectRatio="xMidYMid meet"></svg>
    <div style="font-size:11px;color:var(--dim);margin-top:2px">
      drag on the plot to zoom in on an m/z range &middot; double-click to reset &middot;
      click any peak for details</div>
  </div>

  <div class="q q3">
    <h4>Structure <span id="AMmolname" style="text-transform:none;letter-spacing:0;
      font-weight:400"></span>
      <button id="AMwhole" style="float:right;display:none">whole molecule</button></h4>
    <div id="AMcap" class="cap">click a peak to see how it is formed</div>
    <div id="AMmol"></div>
  </div>

  <div class="q q4">
    <h4>Candidates <span style="text-transform:none;letter-spacing:0;font-weight:400">
      &mdash; click a row to load it</span></h4>
    <div id="AMlist"></div>
    <div class="foot">The bar under each row splits that candidate's spectrum:
      <b style="color:#2f9e44">fragments</b> +
      <b style="color:#1a56c4">diagnostic losses</b> = its score;
      <b style="color:#93b8ee">generic losses</b> are left out.
      Click through the candidates and watch the pale block grow by exactly what the
      first two lose - generic losses come from a fixed library of 9 small neutrals, so
      they reach the same peaks whatever the structure is. Counting them puts every
      candidate on the same total and the ranking disappears.</div>
  </div>

  <div id="AMpop"></div>
</div>

<script>
(function(){
const D = __DATA__;
const R = document.getElementById('AMROOT');
const COL = ['#2f9e44','#1a56c4','#93b8ee','#e0524a'];   // fragment, diagnostic, generic, unexplained
const GREY = '#b9bfc7';
const G = {w:1000,h:430,l:58,r:16,t:18,b:34,base:262};
let cur = 0, lo = null, hi = null, onlyMatched = false, showRed = true, showGrey = true;

const f4 = v => (v==null ? '—' : (+v).toFixed(4));
const esc = s => String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function bounds(){
  let a=Infinity,b=-Infinity;
  D.peaks.forEach(p=>{a=Math.min(a,p[0]);b=Math.max(b,p[0]);});
  const pad=(b-a)*0.02||1; return [a-pad,b+pad];
}
function ticks(a,b,n){
  const raw=(b-a)/n, mag=Math.pow(10,Math.floor(Math.log10(raw)));
  const step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=raw)||10*mag;
  const out=[]; for(let v=Math.ceil(a/step)*step; v<=b; v+=step) out.push(+v.toFixed(6));
  return out;
}
function render(){
  const c = D.cands[cur];
  const [a,b] = (lo==null) ? bounds() : [lo,hi];
  const X = mz => G.l + (mz-a)/(b-a)*(G.w-G.l-G.r);
  const upH = G.base-G.t, dnH = (G.h-G.b)-G.base;
  let s = '';
  // axes
  s += `<line x1="${G.l}" y1="${G.base}" x2="${G.w-G.r}" y2="${G.base}" stroke="#9aa3ad"/>`;
  s += `<line x1="${G.l}" y1="${G.t}" x2="${G.l}" y2="${G.h-G.b}" stroke="#dfe3e8"/>`;
  ticks(a,b,9).forEach(v=>{ const x=X(v); if(x<G.l-1||x>G.w-G.r+1) return;
    s += `<line x1="${x}" y1="${G.h-G.b}" x2="${x}" y2="${G.h-G.b+4}" stroke="#9aa3ad"/>`+
         `<text x="${x}" y="${G.h-G.b+16}" font-size="11" fill="#6b7280" text-anchor="middle">${v}</text>`;});
  [0,50,100].forEach(v=>{ const y=G.base-v/100*upH;
    s += `<line x1="${G.l-4}" y1="${y}" x2="${G.l}" y2="${y}" stroke="#9aa3ad"/>`+
         `<text x="${G.l-7}" y="${y+3.5}" font-size="10.5" fill="#6b7280" text-anchor="end">${v}</text>`;});
  s += `<text x="14" y="${G.t+58}" font-size="11" fill="#6b7280" transform="rotate(-90 14 ${G.t+58})" text-anchor="middle">rel. intensity %</text>`;
  s += `<text x="14" y="${G.base+64}" font-size="11" fill="#6b7280" transform="rotate(-90 14 ${G.base+64})" text-anchor="middle">theoretical</text>`;
  s += `<text x="${G.w-G.r}" y="${G.h-4}" font-size="11" fill="#6b7280" text-anchor="end">m/z</text>`;
  if (D.meta.precursor){ const x=X(D.meta.precursor);
    if(x>G.l&&x<G.w-G.r) s += `<line x1="${x}" y1="${G.t}" x2="${x}" y2="${G.h-G.b}" stroke="#cbd5e1" stroke-dasharray="4 3"/>`+
      `<text x="${x-3}" y="${G.t+10}" font-size="10" fill="#94a3b8" text-anchor="end">precursor</text>`;}
  // theoretical, below
  c.theo.forEach((t,i)=>{ if(onlyMatched && !t[3]) return;
    const x=X(t[0]); if(x<G.l||x>G.w-G.r) return;
    const h = t[3] ? dnH*0.92 : dnH*0.5;
    s += `<line x1="${x}" y1="${G.base}" x2="${x}" y2="${G.base+h}" stroke="${t[3]?'#1a56c4':'#d7dde5'}" stroke-width="${t[3]?1.8:1}"/>`;
    s += `<line class="hit" data-t="1" data-i="${i}" x1="${x}" y1="${G.base}" x2="${x}" y2="${G.base+h}" stroke="transparent" stroke-width="9"/>`;});
  // experimental, above
  D.peaks.forEach((p,i)=>{ const x=X(p[0]); if(x<G.l||x>G.w-G.r) return;
    const e = c.exp[i];
    const isRemoved = p[2]!==0, isRed = !isRemoved && (!e || e.k===3);
    if((isRemoved && !showGrey) || (isRed && !showRed)) return;
    const col = isRemoved ? GREY : (e ? COL[e.k] : COL[3]);
    const y = G.base - Math.max(p[1],0)/100*upH;
    s += `<line x1="${x}" y1="${G.base}" x2="${x}" y2="${y}" stroke="${col}" stroke-width="${p[2]!==0?1.4:2}"/>`;
    s += `<line class="hit" data-t="0" data-i="${i}" x1="${x}" y1="${y-4}" x2="${x}" y2="${G.base}" stroke="transparent" stroke-width="10"/>`;});
  s += `<rect id="AMband" x="0" y="${G.t}" width="0" height="${G.h-G.b-G.t}" fill="#1a56c4" opacity=".10" style="display:none"/>`;
  document.getElementById('AMplot').innerHTML = s;
}
function candList(){
  const seg = p => {                       // task1 | diagnostic | generic | left over
    if(!p) return '';
    const t=p.task1_pct||0, d=p.task2_diagnostic_pct||0, g=p.task2_generic_pct||0;
    const rest=Math.max(0,100-t-d-g);
    return [[t,COL[0]],[d,COL[1]],[g,COL[2]],[rest,'#dfe3e8']]
      .map(x=>`<i style="width:${x[0]}%;background:${x[1]}"></i>`).join('');
  };
  document.getElementById('AMlist').innerHTML =
    '<div class="row hd"><div>name</div><div>SMILES</div><div class="sc">score</div></div>' +
    D.cands.map((c,i)=>
    `<div class="row ${i===cur?'on':''}" data-c="${i}">
       <div class="nm" title="${esc(c.name)}">${esc(c.name)}</div>
       <div class="sm" title="${esc(c.smiles)}">${esc(c.smiles)}</div>
       <div class="sc">${c.score==null?'':c.score.toFixed(1)}</div>
       <div class="bar">${seg(c.parts)}</div>
     </div>`).join('');
}
function wholeMolecule(){
  const c = D.cands[cur];
  document.getElementById('AMmol').innerHTML = c.mol>=0 ? D.svgs[c.mol] : '';
  document.getElementById('AMcap').innerHTML = 'click a peak to see how it is formed';
  document.getElementById('AMwhole').style.display = 'none';
}
function showFormation(i){
  const c = D.cands[cur], s = (c.sites||{})[i];
  const box = document.getElementById('AMmol'), cap = document.getElementById('AMcap');
  box.innerHTML = c.mol>=0 ? D.svgs[c.mol] : '';
  document.getElementById('AMwhole').style.display = 'inline-block';
  const head = '<b>m/z ' + f4(D.peaks[i][0]) + '</b> &mdash; ';
  if (!s){
    cap.innerHTML = head + (D.peaks[i][2]!==0
      ? 'removed before the analysis, so there is nothing to draw'
      : ((c.exp[i] && c.exp[i].k !== 3)
          ? 'a generic small loss (water, CO, CH2 ...) - those have no defined position '
          + 'on the structure, so nothing is highlighted'
          : 'nothing explains this peak, so there is nothing to draw'));
    return;
  }
  cap.innerHTML = head + s.c;
  const svg = box.querySelector('svg'); if(!svg) return;
  const inB = new Set(s.b), inX = new Set(s.x), inA = new Set(s.a);
  svg.querySelectorAll("path[class^='bond-']").forEach(pth=>{
    const n = +pth.getAttribute('class').match(/bond-(\d+)/)[1];
    if (inX.has(n)){ pth.style.stroke='#e0524a'; pth.style.strokeWidth='3.4';
                     pth.style.strokeDasharray='4 3'; }
    else if (inB.has(n)){ pth.style.stroke = (s.k==='l') ? '#e0524a' : COL[0];
                          pth.style.strokeWidth='3.4'; }
    else { pth.style.stroke = (s.k==='l') ? COL[1] : '#dee3e9'; }
  });
  svg.querySelectorAll("path[class^='atom-']").forEach(pth=>{
    const n = +pth.getAttribute('class').match(/atom-(\d+)/)[1];
    if (s.k==='f' && !inA.has(n)) pth.style.opacity = '.28';
  });
}
function loadCand(i){
  cur = i; const c = D.cands[i];
  document.getElementById('AMmolname').textContent = '— ' + c.name;
  candList(); render(); hidePop(); wholeMolecule();
}
function meta(){
  const m = D.meta, d = m.dropped;
  const rows = [
    ['spectrum', m.spectrum], ['annotations', m.annotations],
    ['precursor', (m.precursor?m.precursor.toFixed(4):'—') + (m.adduct?' ['+m.adduct+']':'')],
    ['method', m.method + '  (N = ' + m.n + ')'],
    ['tolerance', m.ppm + ' ppm, floor ' + m.min_da + ' Da'],
    ['noise floor', m.min_rel_int + ' %'],
    ['drop &ge; precursor', m.drop_above ? 'yes' : 'no'],
    ['peaks', d.raw + ' → <b style="color:#22262b">' + d.kept + ' kept</b><br>' +
              '<span style="color:#6b7280">' + d.above_precursor + ' at/above precursor, ' +
              d.below_noise + ' below noise</span>'],
  ];
  document.getElementById('AMmeta').innerHTML =
    rows.map(r=>`<b>${r[0]}</b><span class="mono">${r[1]}</span>`).join('');
  document.getElementById('AMnotes').innerHTML =
    (m.notes&&m.notes.length) ? m.notes.map(n=>`<div class="note">${esc(n)}</div>`).join('') : '';
}
function hidePop(){ document.getElementById('AMpop').style.display='none'; }
function showPop(html, ev){
  const p = document.getElementById('AMpop');
  p.innerHTML = '<span class="x" onclick="this.parentNode.style.display=\'none\'">&times;</span>' + html;
  const r = R.getBoundingClientRect();
  let x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 10;
  p.style.display = 'block';
  x = Math.min(x, r.width - p.offsetWidth - 8);
  y = Math.min(y, r.height - p.offsetHeight - 8);
  p.style.left = Math.max(6,x)+'px'; p.style.top = Math.max(6,y)+'px';
}
function tbl(rows){ return '<table>' + rows.filter(r=>r[1]!=null&&r[1]!=='')
  .map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('') + '</table>'; }

function peakPopup(i, ev){
  const p = D.peaks[i], e = D.cands[cur].exp[i];
  const why = {1:'below the noise floor', 2:'at or above the precursor m/z'};
  const ri = ['rel. intensity', p[1].toFixed(2)+' %'];
  let tag, col, head, body;
  if (p[2] !== 0){
    col = GREY; tag = 'removed'; head = 'm/z ' + f4(p[0]);
    body = tbl([ri, ['reason', why[p[2]]],
      ['', 'dropped before the analysis, so it is out of the score as well']]);
  } else if (!e || e.k === 3){
    col = COL[3]; tag = 'unexplained'; head = 'm/z ' + f4(p[0]);
    body = tbl([ri, ['', 'no theoretical fragment and no neutral loss accounts for this peak']]);
  } else if (e.k === 0){
    col = COL[0]; tag = 'fragment match'; head = e.n || ('fragment ' + e.f);
    body = tbl([['m/z', f4(p[0])], ri, ['formula', e.f], ['adduct', e.a],
      ['theoretical', f4(e.t)],
      ['error', e.p==null?null:(e.p>0?'+':'')+e.p.toFixed(1)+' ppm'],
      ['in score', e.c?'yes':'no']]) +
      (e.s>=0 ? '<div class="st">'+D.svgs[e.s]+'</div>' : '');
  } else {
    col = COL[e.k]; head = 'm/z ' + f4(p[0]);
    tag = e.k===1 ? 'neutral loss (counted)' : 'generic loss (not counted)';
    body = tbl([ri, ['formed by', esc(e.d)]]);
  }
  showPop(`<span class="tag" style="background:${col}">${tag}</span>` +
          `<h5>${esc(head)}</h5>${body}`, ev);
  showFormation(i);                      // and draw how it was formed, bottom-left
}
function theoPopup(i, ev){
  const t = D.cands[cur].theo[i];
  showPop(`<span class="tag" style="background:${t[3]?'#1a56c4':'#94a3b8'}">theoretical fragment ion</span>
     <h5>${esc(t[1])} [${esc(t[2])}]</h5>` +
     tbl([['m/z', f4(t[0])], ['adduct', t[2]],
          ['measured?', t[3] ? 'yes - matches a peak' : 'no - predicted but not observed']]) +
     (t[4]>=0 ? '<div class="st">'+D.svgs[t[4]]+'</div>' : ''), ev);
}

// ---- events ----
document.getElementById('AMplot').addEventListener('click', e=>{
  const h = e.target.closest('.hit'); if(!h) return;
  e.stopPropagation();
  (h.dataset.t==='1' ? theoPopup : peakPopup)(+h.dataset.i, e);
});
document.getElementById('AMlist').addEventListener('click', e=>{
  const r = e.target.closest('.row'); if(r) loadCand(+r.dataset.c);
});
document.getElementById('AMonly').addEventListener('change', e=>{
  onlyMatched = e.target.checked; render();
});
document.getElementById('AMred').addEventListener('change', e=>{
  showRed = e.target.checked; render();
});
document.getElementById('AMgrey').addEventListener('change', e=>{
  showGrey = e.target.checked; render();
});
document.getElementById('AMreset').addEventListener('click', ()=>{lo=hi=null; render();});
document.getElementById('AMwhole').addEventListener('click', wholeMolecule);
R.addEventListener('click', e=>{ if(!e.target.closest('#AMpop') && !e.target.closest('.hit')) hidePop(); });

// drag to zoom
(function(){
  const svg = document.getElementById('AMplot');
  let sx = null;
  const toMz = ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * G.w;
    const [a,b] = (lo==null) ? bounds() : [lo,hi];
    return a + (px - G.l)/(G.w-G.l-G.r)*(b-a);
  };
  const toPx = ev => (ev.clientX - svg.getBoundingClientRect().left) /
                      svg.getBoundingClientRect().width * G.w;
  svg.addEventListener('mousedown', ev=>{ sx = {px:toPx(ev), mz:toMz(ev)}; });
  svg.addEventListener('mousemove', ev=>{
    if(!sx) return; const band = document.getElementById('AMband'); if(!band) return;
    const x = toPx(ev);
    band.setAttribute('x', Math.min(sx.px,x)); band.setAttribute('width', Math.abs(x-sx.px));
    band.style.display = Math.abs(x-sx.px) > 3 ? 'block' : 'none';
  });
  window.addEventListener('mouseup', ev=>{
    if(!sx) return; const a = sx.mz, b = toMz(ev); sx = null;
    if(Math.abs(b-a) > (bounds()[1]-bounds()[0])*0.005){ lo=Math.min(a,b); hi=Math.max(a,b); render(); }
    else { const band=document.getElementById('AMband'); if(band) band.style.display='none'; }
  });
  svg.addEventListener('dblclick', ()=>{lo=hi=null; render();});
})();

meta(); loadCand(0);
})();
</script>
"""


def spectrum_app(R, save=None, all_structures=False, height=790, display=True):
    """The interactive report: one self-contained HTML page for the whole result."""
    import html as _html
    import json as _json
    payload = _json.dumps(_app_payload(R, all_structures), separators=(",", ":"),
                          allow_nan=False)
    page = _APP_TEMPLATE.replace("__DATA__", payload)
    if save is True:
        stem = os.path.splitext(os.path.basename(str(R["spectrum_file"])))[0]
        save = os.path.join(HERE, stem.replace("_ms2_spectrum", "") + "_report.html")
    if save:
        stem = os.path.splitext(os.path.basename(str(R["spectrum_file"])))[0]
        doc = ("<!doctype html><html><head><meta charset='utf-8'>"
               f"<title>{_html.escape(stem)} - annotation report</title></head>"
               f"<body style='margin:0;background:#fff'>{page}</body></html>")
        with open(save, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"wrote {save}  ({len(doc)/1024:.0f} KB)")
    if not display:
        return page
    from IPython.display import HTML
    return HTML(f'<iframe srcdoc="{_html.escape(page, quote=True)}" '
                f'style="width:100%;height:{height}px;border:0" '
                f'sandbox="allow-scripts"></iframe>')
