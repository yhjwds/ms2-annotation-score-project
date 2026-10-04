# Data (not included)

The input data were provided by the supervisor and are not published here yet.
They will be uploaded to this folder once permission to share them has been given.
The notebooks expect this layout:

```
data/
  example_features/            two example lipid features
    FT4679_ms2_spectrum.csv
    FT4679_ipa_annotations.csv
    FT6080_ms2_spectrum.csv
    FT6080_ipa_annotations.csv
  adducts.csv                  optional; otherwise read from the ipaPy2 GitHub repository
  benchmark/                   3,006 MoNA spectra with a known answer
    level1_annotation_summary.csv
    comparison.csv
    MoNA_Agilent_QTOF_MS2_metadata.csv
    level1_annotations/*_MS1annotation.csv
```

All notebook outputs are saved, so the results can be read without running anything.
