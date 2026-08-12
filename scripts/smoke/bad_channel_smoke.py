from sleep_staging.config.settings import load_settings
from sleep_staging.acquisition.loader import SleepEDFLoader
from sleep_staging.preprocessing.bad_channels import BadChannelDetector
from sleep_staging.preprocessing.types import PreprocessedRecording
import json, numpy as np

if __name__ == '__main__':
    settings = load_settings('configs/default.yaml')
    loader = SleepEDFLoader(settings=settings.acquisition)
    paths = loader.discover()
    selected = [p for p in paths if p.name.startswith('SC')][:5]
    reports = []
    for p in selected:
        rec = loader.load_recording(p, preload=True)
        state = PreprocessedRecording.from_sleep_recording(rec, copy=True, preload=True)
        detector = BadChannelDetector()
        state = detector(state)
        raw = state.raw
        data = raw.get_data()
        n_ch, n_times = data.shape
        ch_reports = []
        for idx, ch in enumerate(raw.ch_names):
            vals = data[idx]
            finite = np.isfinite(vals)
            nan_frac = float(1.0 - finite.sum()/float(n_times))
            std = float(np.nanstd(vals))
            p2p = float(np.nanmax(vals) - np.nanmin(vals))
            if finite.sum() > 0:
                maxv = float(np.nanmax(vals))
                minv = float(np.nanmin(vals))
                max_count = int((vals == maxv).sum())
                min_count = int((vals == minv).sum())
                sat_frac = float(max(max_count, min_count))/float(n_times)
            else:
                sat_frac = 1.0
            ch_reports.append({
                'channel': ch,
                'std_V': std,
                'std_uV': std*1e6,
                'p2p_V': p2p,
                'p2p_uV': p2p*1e6,
                'nan_frac': nan_frac,
                'sat_frac': sat_frac,
            })
        reports.append({
            'psg_path': str(p),
            'recording_id': rec.metadata.recording_id,
            'selected_channels': list(raw.ch_names),
            'channels': ch_reports,
            'bad_channels_report': state.extras.get('bad_channels', {}),
        })
    print(json.dumps(reports, indent=2))
