# In silico MS2 fragmentation for scoring IPA metabolite annotations without a spectral library

Yang Jiadong, Francesco Del Carratore. University of Liverpool, Liverpool, UK.
Summer research project (2026).

IPA / ipaPy2 proposes several candidate structures for each LC-MS signal. Checking them
against an MS2 spectrum usually needs a reference library, and many candidates have none.
This project computes the fragments directly from each candidate's structure, scores how
much of the measured MS2 signal each candidate explains, and tests whether that score
ranks the right structure first.

## What is here

| Folder | File | What it is |
|---|---|---|
| `scoring/` | `ms2_scoring.py` | The scoring tool. One call, `explain_spectrum(spectrum, annotations)`, reads two files and scores every candidate. |
| | `ms2_scoring.ipynb` | Worked example on one lipid feature, with all outputs saved. |
| | `FT4679_report.html`, `FT6080_report.html` | Interactive reports. **Download and open in a browser**; click any peak to see the fragment or neutral loss that explains it. |
| `walkthrough/` | `ms2_scoring_stepwise.ipynb` | Step-by-step version: every stage (fragmenting, adducts, peak matching, neutral losses, scoring) in its own cell, with outputs saved. |
| | `ms2_scoring_stepwise.py` | The functions that notebook calls. |
| `benchmark/` | `benchmark_scoring.py` | The scoring method as used for the benchmark (settings frozen: 5 ppm, up to 2 broken bonds, larger neutral-loss library). |
| | `benchmark_run.py` | Runs that method over all 3,006 benchmark spectra and writes the score table. |
| | `benchmark_evaluate.ipynb` | Benchmark on 3,006 MS2 spectra with a known answer: our score against MetFrag, precursor ppm and random ranking (top-k accuracy and RRP), then by compound class and collision energy. Outputs are saved. |
| | `headline.csv`, `figures/` | The main table and figures from that notebook. |
| `data/` | | Not included, see `data/README.md`. |

## Method in short

1. **Clean**: drop noise (below 1 percent) and peaks at or above the precursor.
2. **O1 Fragment**: cut each candidate at BRICS bonds with RDKit and match fragment m/z to peaks (6 adducts; 20 ppm in the examples, 5 ppm in the benchmark).
3. **O2 Neutral loss**: explain leftover peaks by mass gaps that equal a head-group or acyl-chain loss.
4. **O3 Score**: percent of MS2 intensity explained; candidates are ranked by it.
5. **O4 Benchmark**: compare the ranking with MetFrag, ppm and Random on 3,006 spectra.

## Main results

- Every explained peak is traced to a fragment or a neutral loss, shown in the interactive report.
- Overall top-1 accuracy: ours 23 percent, Random 20 percent, MetFrag 34 percent.
- Lipids are the exception: ours matches MetFrag (top-1 31 percent each).

## Running it

Python 3.12 with `rdkit`, `pandas`, `numpy`, `matplotlib` (conda-forge). The notebooks read
their inputs from `../data/`; paths are set in the first code cell.

## References

1. Del Carratore F. et al. (2019) Integrated Probabilistic Annotation. Anal. Chem. 91, 12799-12807.
2. Del Carratore F., Eagles W., Borka J., Breitling R. (2023) ipaPy2: IPA 2.0. Bioinformatics 39(7), btad455.
3. Degen J. et al. (2008) On the art of compiling and using 'drug-like' chemical fragment spaces. ChemMedChem 3, 1503-1507.
4. Ruttkies C. et al. (2016) MetFrag relaunched: incorporating strategies beyond in silico fragmentation. J. Cheminform. 8, 3.
5. Djoumbou Feunang Y. et al. (2016) ClassyFire: automated chemical classification with a comprehensive, computable taxonomy. J. Cheminform. 8, 61.

## AI declaration

See `AI_declaration.txt`.
