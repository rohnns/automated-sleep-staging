# Production pipeline amplitude rejection report

This report was generated from the **production preprocessing pipeline** using `build_default_pipeline(...)`, `preprocess_recording(...)` semantics, `AmplitudeEpochRejector`, and `state.extras["amplitude_reject"]`.

- Recordings processed: 197/197
- Total epochs before rejection: 238317
- Total rejected epochs: 2090
- Total kept epochs: 236227
- Overall rejection percentage: 0.88%

## AASM stage counts before rejection
| Stage | Epochs |
|---|---:|
| W | 70140 |
| N1 | 25175 |
| N2 | 88983 |
| N3 | 19454 |
| REM | 34184 |

## Rejected epochs by AASM stage
| Stage | Rejected | Rejection % of stage |
|---|---:|---:|
| W | 1038 | 1.48% |
| N1 | 111 | 0.44% |
| N2 | 209 | 0.23% |
| N3 | 49 | 0.25% |
| REM | 302 | 0.88% |

## Rejection reasons
| Reason | Count |
|---|---:|
| eog:horizontal:peak_to_peak | 1077 |
| eeg:Pz-Oz:peak_to_peak | 573 |
| eeg:Fpz-Cz:peak_to_peak | 485 |