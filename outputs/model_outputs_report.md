# Phase 4 Final Results Report

## Experiment scope
- Dataset: Sleep-EDF Expanded, SC cohort
- Split: subject-wise, seed 42
- Controlled channels for the initial bake-off: Fpz-Cz only
- Models / representations: RawCNN1D, BandPowerMLP, STFTCNN2D
- Test set: untouched held-out subjects only
- IGNORE epochs: excluded from loss and metrics

## Subject-wise split audit

### Partition sizes
- Train: 55 subjects
- Validation: 12 subjects
- Test: 11 subjects

### Exact subject lists

**Train (55)**
SC433, SC451, SC441, SC428, SC461, SC425, SC400, SC460, SC421, SC437, SC418, SC443, SC426, SC457, SC459, SC404, SC427, SC424, SC417, SC407, SC429, SC473, SC482, SC481, SC458, SC464, SC405, SC401, SC474, SC444, SC402, SC446, SC463, SC475, SC420, SC477, SC453, SC432, SC440, SC423, SC480, SC449, SC415, SC409, SC431, SC452, SC416, SC445, SC447, SC456, SC462, SC403, SC434, SC430, SC410

**Validation (12)**
SC470, SC476, SC455, SC450, SC406, SC438, SC472, SC411, SC471, SC442, SC422, SC419

**Test (11)**
SC435, SC465, SC448, SC412, SC454, SC414, SC413, SC467, SC436, SC466, SC408

### Leakage check
- Train ∩ Validation = ∅
- Train ∩ Test = ∅
- Validation ∩ Test = ∅
- Conclusion: zero subject leakage

### Epoch counts
- Train: 136,539 total epochs, 136,243 supervised
- Validation: 31,102 total epochs, 30,879 supervised
- Test: 28,353 total epochs, 28,347 supervised

## Aggregate test-set results

| Representation | Accuracy | Macro-F1 | Cohen's kappa |
|---|---:|---:|---:|
| Raw | 0.8219 | 0.7754 | 0.7669 |
| BandPower | 0.7928 | 0.6964 | 0.7275 |
| STFT | 0.7948 | 0.7339 | 0.7236 |

### Best overall
- Best accuracy: **Raw**
- Best macro-F1: **Raw**
- Best kappa: **Raw**

## Per-class metrics

### Raw
| Metric | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| Precision | 0.9400 | 0.4884 | 0.7087 | 0.9814 | 0.7397 |
| Recall | 0.8731 | 0.6176 | 0.9008 | 0.7543 | 0.8244 |
| F1 | 0.9053 | 0.5455 | 0.7933 | 0.8530 | 0.7798 |

Confusion matrix:
```text
[[282  12   1   0  28]
 [ 18  42   1   1   6]
 [  0  18 236   4   4]
 [  0   6  80 264   0]
 [  0   8  15   0 108]]
```

Normalized confusion matrix:
```text
[[0.8731 0.0372 0.0031 0.0000 0.0867]
 [0.2647 0.6176 0.0147 0.0147 0.0882]
 [0.0000 0.0687 0.9008 0.0153 0.0153]
 [0.0000 0.0171 0.2286 0.7543 0.0000]
 [0.0000 0.0611 0.1145 0.0000 0.8244]]
```

### BandPower
| Metric | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| Precision | 0.9010 | 0.4118 | 0.6918 | 0.9850 | 0.5895 |
| Recall | 0.8731 | 0.2059 | 0.8740 | 0.7486 | 0.8550 |
| F1 | 0.8868 | 0.2745 | 0.7723 | 0.8506 | 0.6978 |

Confusion matrix:
```text
[[282  15   1   0  25]
 [ 27  14   1   0  26]
 [  2   3 229   4  24]
 [  2   0  83 262   3]
 [  0   2  17   0 112]]
```

Normalized confusion matrix:
```text
[[0.8731 0.0464 0.0031 0.0000 0.0774]
 [0.3971 0.2059 0.0147 0.0000 0.3824]
 [0.0076 0.0115 0.8740 0.0153 0.0916]
 [0.0057 0.0000 0.2371 0.7486 0.0086]
 [0.0000 0.0153 0.1298 0.0000 0.8550]]
```

### STFT
| Metric | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| Precision | 0.9767 | 0.4508 | 0.8612 | 0.6301 | 0.6425 |
| Recall | 0.8771 | 0.4831 | 0.7437 | 0.9192 | 0.8539 |
| F1 | 0.9242 | 0.4664 | 0.7981 | 0.7477 | 0.7333 |

Confusion matrix:
```text
[[9509  986   54   16  276]
 [ 192 1411  594   28  696]
 [  20  539 6707 1187  566]
 [   3    9  167 2117    7]
 [  12  185  266   12 2777]]
```

Normalized confusion matrix:
```text
[[0.8771 0.0910 0.0050 0.0015 0.0255]
 [0.1068 0.4831 0.3303 0.0156 0.0642]
 [0.0022 0.0594 0.7437 0.1316 0.0630]
 [0.0013 0.0039 0.0737 0.9192 0.0019]
 [0.0033 0.0511 0.0734 0.0033 0.8539]]
```

## Comparison: Raw vs BandPower vs STFT

- **Raw** is the strongest overall on this baseline run.
- **STFT** is second-best overall and is notably strong for N3 recall/F1.
- **BandPower** is the weakest overall, especially on N1, where recall and F1 are substantially lower than Raw/STFT.

### Main confusions
- All three models struggle most with **N1**.
- Major confusions are:
  - **N1 ↔ W**
  - **N1 ↔ N2**
  - **N2 ↔ N3** for STFT and BandPower in some cases
  - REM is generally better separated than N1, but still confuses with W/N2 in some epochs

### Interpretation
- The class imbalance means accuracy is dominated by W and N2, so macro-F1 is more informative.
- Raw preserves the most information for this controlled comparison and performs best overall.
- STFT retains enough time-frequency structure to be competitive and is particularly useful for sleep depth-related structure like N3.
- BandPower compresses the signal the most and appears to lose discriminative temporal detail, which likely hurts N1 in particular.

## Exported output artifacts
For each held-out test PSG, outputs were generated under:
- `outputs/model_outputs/<recording_stem>/raw/`
- `outputs/model_outputs/<recording_stem>/bandpower/`
- `outputs/model_outputs/<recording_stem>/time_frequency/`

Each contains:
- `predictions.csv`
- `hypnogram.png`
- `summary.json`

Top-level:
- `outputs/model_outputs/manifest.json`
