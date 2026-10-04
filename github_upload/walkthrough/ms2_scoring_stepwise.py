"""
ms2_scoring_stepwise.py
Step-by-step version of the scoring pipeline, used by ms2_scoring_stepwise.ipynb.

Each stage is a separate function so the notebook can show it: load the spectrum,
cut every candidate at BRICS bonds (RDKit), turn fragments into ions (6 adducts),
match them to peaks, explain leftover peaks by neutral losses, and score each candidate.
The SMILES in the annotation file are neutral compounds, not ions.
"""
import os
import re
import math
from itertools import combinations
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Draw, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog("rdApp.*")

PROTON, ELECTRON = 1.0072765, 0.0005486
TOL_PPM, TOL_MIN = 20.0, 0.005


DEFAULT_METHOD = "BRICSDecompose"
MAX_BREAKS = 2
MIN_REL_INT = 1.0


DROP_ABOVE_PRECURSOR = True


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "example_features")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)
_METHOD_FILE = os.path.join(HERE, ".last_method")


def set_tolerance(ppm, min_da=None):
    """Set the m/z matching tolerance used by ALL peak matching (Task 1 & 2)."""
    global TOL_PPM, TOL_MIN
    TOL_PPM = float(ppm)
    if min_da is not None:
        TOL_MIN = float(min_da)
    return TOL_PPM, TOL_MIN


ADDUCTS_URL = "https://raw.githubusercontent.com/francescodc87/ipaPy2/main/DB/adducts.csv"
ADDUCTS_LOCAL = os.path.join(HERE, "..", "data", "adducts.csv")

FRAGMENT_ADDUCTS = ["M+H", "M+", "M+Na", "M+2H", "2M+H", "M+NH4"]
_ADDUCT_CACHE = None


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
                mass=float(r["Mass"])) for _, r in df.iterrows()]
    missing = [n for n in names if n not in set(df["name"])]
    note = f"  (missing: {', '.join(missing)})" if missing else ""
    print(f"[adducts] {len(out)} loaded from {src}: "
          f"{', '.join(a['name'] for a in out)}{note}")
    return out


def get_fragment_adducts():
    """The 6 selected adducts, loaded once and cached for the whole session."""
    global _ADDUCT_CACHE
    if _ADDUCT_CACHE is None:
        _ADDUCT_CACHE = load_adducts()
    return _ADDUCT_CACHE


def adduct_mz(neutral_mass, adduct):
    """Theoretical ion m/z of a neutral mass under one adduct dict:"""
    return (adduct["mult"] * neutral_mass) / adduct["charge"] + adduct["mass"]


def theoretical_matrix(frags, adducts=None):
    """Readable view of fragment_ions: ONE row per fragment, ONE column per adduct,"""
    ions = fragment_ions(frags, adducts)
    df = pd.DataFrame(ions)
    df["fragment"] = df["formula"] + "  " + df["frag"].str.slice(0, 22) + \
        df["frag"].str.len().gt(22).map({True: "...", False: ""})
    mat = df.pivot_table(index="fragment", columns="adduct", values="mz")
    cols = [a for a in FRAGMENT_ADDUCTS if a in mat.columns] + \
           [c for c in mat.columns if c not in FRAGMENT_ADDUCTS]
    return mat[cols].round(4)


def resolve_method(method=None):
    """Choose the fragmentation method."""
    if method:
        with open(_METHOD_FILE, "w") as f:
            f.write(method)
        return method
    if os.path.exists(_METHOD_FILE):
        return open(_METHOD_FILE).read().strip() or DEFAULT_METHOD
    return DEFAULT_METHOD


def cap(frag_smiles):
    """Cap BRICS dummy atoms with H (so a fragment becomes a real molecule)."""
    return Chem.MolFromSmiles(re.sub(r"\[\d*\*\]|\*", "[H]", frag_smiles))


def _combinatorial_pieces(mol, max_breaks):
    """Break combinations of 1..max_breaks BRICS bonds; return dummy-labelled pieces."""
    bonds = [mol.GetBondBetweenAtoms(a, b).GetIdx()
             for (a, b), _ in BRICS.FindBRICSBonds(mol)]
    pieces = set()
    for n in range(1, max_breaks + 1):
        for combo in combinations(bonds, n):
            fm = Chem.FragmentOnBonds(mol, combo, addDummies=True)
            pieces.update(Chem.MolToSmiles(fm).split("."))
    return pieces


def brics_cut(mol, method=DEFAULT_METHOD, max_breaks=MAX_BREAKS):
    """Return the raw fragment SMILES (with dummy atoms) for the chosen method."""
    if method == "BRICSDecompose":
        return sorted(BRICS.BRICSDecompose(mol))
    if method == "FragmentOnBonds":
        return sorted(_combinatorial_pieces(mol, max_breaks))
    raise ValueError(f"unknown method: {method}")


def brics_fragments(mol, method=DEFAULT_METHOD, max_breaks=MAX_BREAKS):
    """Cut -> cap -> mass."""
    out = {}
    for piece in brics_cut(mol, method, max_breaks):
        m = cap(piece)
        if m is None:
            continue
        smi = Chem.MolToSmiles(m)
        out[smi] = dict(smiles=smi,
                        formula=rdMolDescriptors.CalcMolFormula(m),
                        mass=ExactMolWt(m),
                        charge=Chem.GetFormalCharge(m))
    return list(out.values())


def fragment_ions(frags, adducts=None):
    """Neutral fragment -> theoretical ion m/z, one entry per selected adduct."""
    if adducts is None:
        adducts = get_fragment_adducts()
    ions = []
    for f in frags:
        if f["charge"] != 0:


            ions.append(dict(mz=f["mass"] + (1 - f["charge"]) * PROTON - ELECTRON,
                             ion_type="intrinsic", adduct="intrinsic",
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


def _guess_adduct(precursor_mz, neutral_mass):
    """Name the adduct of the precursor by matching (precursor - neutral) to _ADDUCTS."""
    d = precursor_mz - neutral_mass
    label, add = min(_ADDUCTS, key=lambda a: abs(d - a[1]))
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


def _annotate_special(rows, mol, precursor_mz):
    """Give precursor and isotope peaks a real annotation (not just a tag):"""
    neutral = ExactMolWt(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    for r in rows:
        if r["special"] == "precursor" and precursor_mz is not None:
            ion_type, theo = _guess_adduct(precursor_mz, neutral)
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


def analyze(peaks, smiles, method=DEFAULT_METHOD, max_breaks=MAX_BREAKS, precursor_mz=None):
    """Full pipeline for ONE candidate: cut -> cap -> mass -> [M+H]+ -> match peaks."""
    mol = Chem.MolFromSmiles(smiles)
    frags = brics_fragments(mol, method, max_breaks)
    ions = fragment_ions(frags)
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
    _annotate_special(rows, mol, precursor_mz)
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


def load(feat, min_rel_int=None, drop_above_precursor=None):
    """Read the annotation table and MS2 peak list for a feature (e.g. 'FT4679')."""


    if min_rel_int is None:
        min_rel_int = MIN_REL_INT
    if drop_above_precursor is None:
        drop_above_precursor = DROP_ABOVE_PRECURSOR
    ann = pd.read_csv(os.path.join(DATA, f"{feat}_ipa_annotations.csv"))
    ms2 = pd.read_csv(os.path.join(DATA, f"{feat}_ms2_spectrum.csv")).sort_values("mz")
    if drop_above_precursor and "precursor_mz" in ms2.columns:
        ms2 = ms2[ms2["mz"] < float(ms2["precursor_mz"].iloc[0])]
    ms2 = ms2[ms2["rel_intensity"] >= min_rel_int]
    peaks = list(ms2[["mz", "rel_intensity"]].itertuples(index=False, name=None))
    return ann, peaks


def read_precursor(feat):
    """Precursor (parent-ion) m/z from the MS2 file, used to tag the surviving"""
    ms2 = pd.read_csv(os.path.join(DATA, f"{feat}_ms2_spectrum.csv"))
    return float(ms2["precursor_mz"].iloc[0]) if "precursor_mz" in ms2.columns else None


def analyze_all(feat, method=DEFAULT_METHOD, max_breaks=MAX_BREAKS,
                adduct=None, score=True, include_generic=False):
    """Run analyze() over EVERY candidate in the annotation file and SCORE each one"""
    ann, peaks = load(feat)
    prec = read_precursor(feat)
    if adduct is None and "adduct" in ann.columns:
        adduct = str(ann["adduct"].iloc[0])
    cands = []
    for _, r in ann.iterrows():
        res = analyze(peaks, str(r["smiles"]), method, max_breaks, precursor_mz=prec)
        res["name"], res["id"], res["adduct"] = r["name"], r["id"], adduct
        if score:
            parts = candidate_score(res, adduct, include_generic)
            res["score"], res["score_parts"] = parts["score"], parts
            res["score_generic"] = include_generic
        cands.append(res)
    cands.sort(key=lambda c: (c.get("score", c["coverage"]["pct_intensity"]),
                              c["coverage"]["pct_intensity"]), reverse=True)
    return cands


def _find(cands, name):
    for c in cands:
        if name in (c["name"], c["id"]):
            return c
    raise KeyError(f"candidate not found: {name}")


def coverage_table(cands):
    """Level-1 view: one row per candidate with its coverage, plus the overall score"""
    rows = []
    for c in cands:
        row = dict(name=c["name"], id=c["id"])
        if "score" in c:
            row["score"] = c["score"]
        row.update(c["coverage"])
        rows.append(row)
    return pd.DataFrame(rows)


def peak_matches(cands, name):
    """Level-2 view for one candidate (by name or id): every experimental peak and,"""
    c = _find(cands, name)
    return pd.DataFrame(c["peaks_table"]).sort_values("rel_int", ascending=False).reset_index(drop=True)


_ANNOT_COLS = ["exp_mz", "rel_int", "kind", "name", "formula", "best_adduct",
               "theo_mz", "ppm", "n_hits", "alt_adducts", "n_frag", "frag_smiles"]


def annotated_table(cands, name):
    """The annotation table: every peak we CAN explain, with its identity. Covers"""
    c = _find(cands, name)
    df = pd.DataFrame(c["peaks_table"])
    df = df[df["kind"] != ""].copy()
    return df[_ANNOT_COLS].sort_values("rel_int", ascending=False).reset_index(drop=True)


def unexplained_table(cands, name):
    """The peaks NO theoretical fragment explains (precursor / isotope peaks removed)."""
    c = _find(cands, name)
    df = pd.DataFrame(c["peaks_table"])
    df = df[(~df["matched"]) & (df["special"] == "")].copy()
    return df[["exp_mz", "rel_int"]].sort_values("rel_int", ascending=False).reset_index(drop=True)


def adduct_breakdown(c):
    """Output ③: for one candidate, how many fragment peaks each adduct explained and"""
    total = sum(r["rel_int"] for r in c["peaks_table"]) or 1
    agg = {}
    for r in c["peaks_table"]:
        if r["kind"] != "fragment":
            continue
        a = r["best_adduct"] or "intrinsic"
        n, s = agg.get(a, (0, 0.0))
        agg[a] = (n + 1, s + r["rel_int"])
    order = FRAGMENT_ADDUCTS + ["intrinsic"]
    rows = [dict(adduct=a, n_peaks=agg[a][0], pct_intensity=round(100 * agg[a][1] / total, 1))
            for a in order if a in agg]
    return pd.DataFrame(rows)


def style_annotated(cands, name):
    """### 9 with light styling (returns a pandas Styler): intensity drawn as a bar,"""
    df = annotated_table(cands, name).drop(columns=["frag_smiles"], errors="ignore")
    sty = (df.style
           .format({"exp_mz": "{:.4f}", "theo_mz": "{:.4f}", "ppm": "{:+.1f}",
                    "rel_int": "{:.1f}"}, na_rep="")
           .set_properties(subset=["ppm"], color="#888780"))
    if "n_hits" in df.columns:
        sty = sty.apply(lambda s: ["background-color: #FAEEDA" if (v or 0) > 1 else ""
                                   for v in s], subset=["n_hits"])
    return sty


def explained_peaks(cands):
    """Peaks that AT LEAST ONE candidate can explain with a theoretical fragment."""
    peak_mzs = [r["exp_mz"] for r in cands[0]["peaks_table"]]
    keep = [mz for mz in peak_mzs
            if any(any(match(mz, io["mz"]) for io in c["ions"]) for c in cands)]
    return sorted(keep)


def candidate_peak_matrix(cands, peak_mzs, value="check"):
    """Cross-candidate view (B3): rows = candidates, columns = chosen peaks,"""
    rows = []
    for c in cands:
        row = {"candidate": c["name"] or c["id"]}
        for mz in peak_mzs:
            hit = next((io for io in c["ions"] if match(mz, io["mz"])), None)
            if value == "formula":
                row[f"{mz:.3f}"] = hit["formula"] if hit else ""
            else:
                row[f"{mz:.3f}"] = "✓" if hit else ""
        rows.append(row)
    return pd.DataFrame(rows)


def draw_annotated_peaks(cands, name, save=True):
    """Draw ONLY the fragments that explain a peak, each captioned with the"""
    c = _find(cands, name)
    seen = {}
    for r in sorted(c["peaks_table"], key=lambda r: -r["rel_int"]):
        if r["matched"] and r["frag_smiles"] not in seen:
            seen[r["frag_smiles"]] = r
    mols = [Chem.MolFromSmiles(s) for s in seen]
    legs = [f'{r["exp_mz"]:.3f}  {r["name"] or r["formula"]}' for r in seen.values()]
    img = Draw.MolsToGridImage(mols, legends=legs, molsPerRow=3, subImgSize=(260, 200))
    if save and mols:
        safe = re.sub(r"[^\w]+", "_", name).strip("_")
        path = os.path.join(FIGDIR, f"{safe}_annotated_peaks.png")
        if hasattr(img, "save"):
            img.save(path)
        else:
            with open(path, "wb") as fh:
                fh.write(img.data)
    return img


def draw_fragments(cands, name, save=True):
    """Draw the BRICS fragments of one candidate (tutorial: Draw.MolsToGridImage)."""
    c = _find(cands, name)
    mols = [Chem.MolFromSmiles(f["smiles"]) for f in c["frags"]]
    legs = [f["formula"] for f in c["frags"]]
    img = Draw.MolsToGridImage(mols, legends=legs, molsPerRow=3, subImgSize=(240, 180))
    if save:
        safe = re.sub(r"[^\w]+", "_", name).strip("_")
        path = os.path.join(FIGDIR, f"{safe}_fragments.png")
        if hasattr(img, "save"):
            img.save(path)
        else:
            with open(path, "wb") as fh:
                fh.write(img.data)
    return img


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
]
_LOSS_RULES = [(Chem.MolFromSmarts(s), losses) for s, losses in _LOSS_RULES_SMARTS]

_ACYL_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
_ACID_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")


def _acid_label(acid_mol):
    """'16:0' style label (carbons : C=C in the chain) from an RCOOH fragment."""
    c = _parse_formula(rdMolDescriptors.CalcMolFormula(acid_mol))
    nc, nh, nn = c.get("C", 0), c.get("H", 0), c.get("N", 0)
    dou = (2 * nc + 2 + nn - nh) / 2
    return f"{nc}:{int(dou) - 1}"


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
            label = _acid_label(acid)
            out[label] = dict(mass=acid_mass,
                              formula=rdMolDescriptors.CalcMolFormula(acid),
                              name=f"fatty acid {label} (RCOOH)", source="acyl")
    losses = list(out.values())
    for a in list(out.values()):
        losses.append(dict(mass=a["mass"] - formula_mass("H2O"),
                           formula=a["formula"], source="acyl",
                           name=a["name"].replace("(RCOOH)", "as ketene (-H2O)")))
    return losses


def candidate_losses(mol, adduct=None):
    """Library 2 as an ALGORITHM: all neutral losses this precursor structure can"""
    losses = [dict(mass=formula_mass(f), formula=f, name=nm, source="universal")
              for f, nm in _UNIVERSAL]
    for query, rules in _LOSS_RULES:
        if query is not None and mol.HasSubstructMatch(query):
            losses += [dict(mass=formula_mass(f), formula=f, name=nm, source="headgroup")
                       for f, nm in rules]
    acyls = acyl_losses(mol)
    losses += acyls
    if adduct and "NH4" in adduct:
        nh3 = formula_mass("NH3")
        for a in acyls:
            losses.append(dict(mass=nh3 + a["mass"], formula=f'NH3+{a["formula"]}',
                               name=f'NH3 + {a["name"]}', source="adduct+acyl"))
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
    for label, add in _ADDUCTS:
        if re.sub(r"[^A-Za-z0-9]", "", label) == norm:
            return neutral + add
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
    if source == "headgroup":
        return "head-group"
    if formula in _SPACING_FORMULAS:
        return "series spacing"
    return "small loss"


def _adduct_lookup():
    """{adduct name -> its dict} for back-calculating neutral cores."""
    return {a["name"]: a for a in get_fragment_adducts()}


def _neutral_core(mz, adduct_name, lut):
    """A peak's neutral mass, obtained by stripping its adduct; None if the adduct is"""
    a = lut.get(adduct_name)
    return None if a is None else (mz - a["mass"]) * a["charge"] / a["mult"]


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


    other_mz = {a["name"]: adduct_mz(M, a) for a in get_fragment_adducts()}
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
    df = pd.DataFrame(rows).sort_values("rel_int", ascending=False).reset_index(drop=True)
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
    df = pd.DataFrame(rows)
    if len(df):
        df["_r"] = df["category"].map(_CAT_RANK).fillna(9)
        df = df.sort_values(["_r", "mz_hi"], ascending=[True, False]).drop(columns="_r").reset_index(drop=True)
    if matched_only and len(df):
        df = df[df["matched"]].reset_index(drop=True)
    return df


_DIAGNOSTIC_CATS = {"acyl bridge", "head-group"}


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


def tg_acyl_readout(cands, adduct):
    """B2 - candidate discrimination. A big fragment = precursor - (NH3 + fatty acid),"""
    per, diag = {}, set()
    for c in cands:
        pl = precursor_losses(c, adduct)
        acyl = pl[pl["matched"] & pl["source"].isin(["acyl", "adduct+acyl"])]
        hits = {round(r["exp_mz"], 2): r["loss"].replace("NH3 + fatty acid ", "")
                                               .replace("fatty acid ", "")
                                               .replace(" (RCOOH)", "")
                for _, r in acyl.iterrows()}
        per[c["name"] or c["id"]] = (round(acyl["rel_int"].sum(), 1), hits)
        diag.update(hits)
    diag = sorted(diag)
    rows = [dict(candidate=name, n_peaks=len(hits), intensity=inten,
                 **{f"{mz:.2f}": hits.get(mz, "") for mz in diag})
            for name, (inten, hits) in per.items()]
    return (pd.DataFrame(rows)
            .sort_values(["n_peaks", "intensity"], ascending=False)
            .reset_index(drop=True))


def candidate_score(c, adduct, include_generic=False):
    """Score ONE candidate: the share of the total measured MS2 intensity this"""
    _, resc = loss_rescue(c, adduct)
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
