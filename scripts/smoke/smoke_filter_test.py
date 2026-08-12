from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve() / 'src'))
import mne
import numpy as np
from sleep_staging.preprocessing import ChannelSelector, BadChannelDetector, ReferenceTransform, SignalFilter
from sleep_staging.preprocessing.types import PreprocessedRecording
# Do not rely on a top-level loader; use mne and wrap when needed
# from sleep_staging.acquisition import load_recording_from_path

# Locate dataset
data_root = Path('D:/SleepEDFX')
cassette = data_root / 'sleep-cassette'
if not cassette.exists():
    cassette = data_root

# collect candidate EDF files that look like SC recordings (prefix SC)
files = sorted([p for p in cassette.rglob('*.edf') if p.name.upper().startswith('SC') and 'PSG' in p.name.upper()])
if not files:
    print('No SC files found under', cassette)
    raise SystemExit(1)

files = files[:5]
print('Selected files:', *[f.name for f in files], sep='\n- ')

results = []
for f in files:
    print('\n\nProcessing', f)
    # load via load_recording_from_path if available, else use mne
    try:
        rec = load_recording_from_path(f, preload=True)
    except Exception as e:
        print('load_recording_from_path failed, falling back to mne.io.read_raw_edf:', e)
        rec = mne.io.read_raw_edf(str(f), preload=True, verbose='ERROR')
        # wrap into minimal PreprocessedRecording
        from sleep_staging.acquisition.dataclasses import RecordingMetadata, ChannelInfo
        ch_names = rec.info['ch_names']
        ch_types = tuple(rec.get_channel_types())
        channels = tuple(ChannelInfo(name=n, ch_type=t, unit='V', sampling_frequency=rec.info['sfreq']) for n,t in zip(ch_names,ch_types))
        metadata = RecordingMetadata(subject_id='unknown', recording_id=f.stem, study='SC', sampling_frequency=rec.info['sfreq'], duration_sec=rec.n_times/rec.info['sfreq'], n_channels=len(ch_names), channel_names=tuple(ch_names), channel_types=ch_types, channels=channels, units={n:'V' for n in ch_names}, reference='orig', montage=None, meas_date=None, psg_path=f, hypnogram_path=Path(''), n_annotations=0)
        from sleep_staging.preprocessing.types import PreprocessedRecording
        state = PreprocessedRecording(raw=rec, metadata=metadata)
    else:
        # if loaded via project loader, it might already return PreprocessedRecording
        if isinstance(rec, PreprocessedRecording):
            state = rec
        else:
            # assume it's an mne Raw
            raw = rec
            from sleep_staging.acquisition.dataclasses import RecordingMetadata, ChannelInfo
            ch_names = raw.info['ch_names']
            ch_types = tuple(raw.get_channel_types())
            channels = tuple(ChannelInfo(name=n, ch_type=t, unit='V', sampling_frequency=raw.info['sfreq']) for n,t in zip(ch_names,ch_types))
            metadata = RecordingMetadata(subject_id='unknown', recording_id=f.stem, study='SC', sampling_frequency=raw.info['sfreq'], duration_sec=raw.n_times/raw.info['sfreq'], n_channels=len(ch_names), channel_names=tuple(ch_names), channel_types=ch_types, channels=channels, units={n:'V' for n in ch_names}, reference='orig', montage=None, meas_date=None, psg_path=f, hypnogram_path=Path(''), n_annotations=0)
            state = PreprocessedRecording(raw=raw, metadata=metadata)

    # record pre stats
    pre_data = state.raw.get_data().copy()
    sfreq = state.sampling_frequency
    n_samples = state.raw.n_times
    ch_names = state.raw.info['ch_names']
    ch_types = state.raw.get_channel_types()

    pre_stats = {}
    for idx, ch in enumerate(ch_names):
        arr = pre_data[idx]
        pre_stats[ch] = {
            'type': ch_types[idx],
            'std': float(np.std(arr)),
            'ptp': float(np.ptp(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'nan_frac': float(np.count_nonzero(~np.isfinite(arr))/arr.size),
        }

    # run transforms: ChannelSelector (defaults). If default name-based selection fails
    # (real EDF uses prefixed names like 'EEG Fpz-Cz'), fall back to type-based selection
    from sleep_staging.preprocessing.exceptions import MissingChannelsError
    try:
        cs = ChannelSelector()
        state = cs(state)
    except MissingChannelsError as e:
        print('Default ChannelSelector failed:', e)
        print('Falling back to type-based ChannelSelector (eeg,eog,emg)')
        cs = ChannelSelector(names=None, types=('eeg','eog','emg'), require_all_names=False)
        state = cs(state)
    bcd = BadChannelDetector()
    state = bcd(state)
    ref = ReferenceTransform(mode='original')
    state = ref(state)
    filt = SignalFilter()
    state = filt(state)

    post_data = state.raw.get_data().copy()
    post_stats = {}
    for idx, ch in enumerate(state.raw.info['ch_names']):
        arr = post_data[idx]
        post_stats[ch] = {
            'type': state.raw.get_channel_types()[idx],
            'std': float(np.std(arr)),
            'ptp': float(np.ptp(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'nan_frac': float(np.count_nonzero(~np.isfinite(arr))/arr.size),
        }

    # helper: find channel by substring
    def find_channel(sub):
        for name in ch_names:
            if sub in name:
                return name
        for name in state.raw.ch_names:
            if sub in name:
                return name
        return None

    # PSD check: compute Welch before/after for selected channels of interest
    from scipy.signal import welch
    def compute_psd(x):
        nperseg = 4096 if n_samples>=4096 else n_samples
        f, P = welch(x, fs=sfreq, nperseg=nperseg)
        return f, P

    psd_report = {}
    for logical_name in ['Fpz-Cz','Pz-Oz','horizontal','submental']:
        ch = find_channel(logical_name)
        if ch is not None and ch in pre_stats:
            idx = ch_names.index(ch)
            f_pre, P_pre = compute_psd(pre_data[idx])
            pidx = state.raw.ch_names.index(ch)
            f_post, P_post = compute_psd(post_data[pidx])
            # compute simple attenuation measures
            psd_report[logical_name] = dict(
                ch_name=ch,
                f=f_pre.tolist(),
                pre=P_pre.tolist(),
                post=P_post.tolist(),
            )
        else:
            psd_report[logical_name] = None

    results.append(dict(
        file=str(f),
        sfreq=sfreq,
        n_samples_before=int(n_samples),
        n_samples_after=int(state.raw.n_times),
        ch_names_before=ch_names,
        ch_names_after=state.raw.info['ch_names'],
        pre_stats=pre_stats,
        post_stats=post_stats,
        extras_filter=state.extras.get('filter', {}),
        bads=state.extras.get('bad_channels', {}),
        psd=psd_report,
    ))

    # PSD check: compute Welch before/after for selected channels of interest
    from scipy.signal import welch
    def compute_psd(x):
        nperseg = 4096 if n_samples>=4096 else n_samples
        f, P = welch(x, fs=sfreq, nperseg=nperseg)
        return f, P

    psd_report = {}
    for name in ['Fpz-Cz','Pz-Oz','horizontal','submental']:
        if name in ch_names:
            idx = ch_names.index(name)
            f_pre, P_pre = compute_psd(pre_data[idx])
            pidx = state.raw.ch_names.index(name)
            f_post, P_post = compute_psd(post_data[pidx])
            # compute attenuation metrics: sum power outside band vs inside band
            psd_report[name] = dict(f=f_pre.tolist(), pre=P_pre.tolist(), post=P_post.tolist())
        else:
            psd_report[name] = None

    results.append(dict(
        file=str(f),
        sfreq=sfreq,
        n_samples_before=n_samples,
        n_samples_after=state.raw.n_times,
        ch_names_before=ch_names,
        ch_names_after=state.raw.info['ch_names'],
        pre_stats=pre_stats,
        post_stats=post_stats,
        extras_filter=state.extras.get('filter', {}),
        bads=state.extras.get('bad_channels', {}),
        psd=psd_report,
    ))

# Print summary
for r in results:
    def find_channel_in(r, sub):
        for name in r['ch_names_before']:
            if sub in name:
                return name
        for name in r['ch_names_after']:
            if sub in name:
                return name
        return None

    print('\n---')
    print('File:', r['file'])
    print('Sampling frequency:', r['sfreq'])
    print('Samples before/after:', r['n_samples_before'], '/', r['n_samples_after'])
    print('Channels before:', r['ch_names_before'])
    print('Channels after:', r['ch_names_after'])
    print('\nPer-channel stats (pre -> post) for staging channels:')
    for logical in ['Fpz-Cz','Pz-Oz','horizontal','submental']:
        actual = find_channel_in(r, logical)
        print('\nLogical channel:', logical)
        if actual is None:
            print('  Not present in recording')
            continue
        pre = r['pre_stats'].get(actual)
        post = r['post_stats'].get(actual)
        print('  Actual channel name:', actual)
        print('  Type:', pre['type'])
        print('  STD (pre -> post):', pre['std'], '->', post['std'])
        print('  P2P (pre -> post):', pre['ptp'], '->', post['ptp'])
        print('  Min/Max (pre -> post):', (pre['min'], pre['max']), '->', (post['min'], post['max']))
        print('  NaN frac (pre -> post):', pre['nan_frac'], '->', post['nan_frac'])
        # Filter config for this channel
        cfg = r['extras_filter'].get('per_channel', {}).get(actual)
        print('  Applied band:', cfg)
        # PSD attenuation quick checks
        psd = r['psd'].get(logical)
        if psd is not None:
            f = np.array(psd['f'])
            preP = np.array(psd['pre'])
            postP = np.array(psd['post'])
            # compute power sums in bands
            eeg_band = (f>=0.5)&(f<=30)
            low_band = f<0.5
            high_band = f>30
            print('  PSD power ratio: low/total pre=', float(preP[low_band].sum()/preP.sum()) if preP.sum()>0 else None,
                  'post=', float(postP[low_band].sum()/postP.sum()) if postP.sum()>0 else None)
            print('                 high/total pre=', float(preP[high_band].sum()/preP.sum()) if preP.sum()>0 else None,
                  'post=', float(postP[high_band].sum()/postP.sum()) if postP.sum()>0 else None)
    print('\nBad channels marked:', r['bads']['all'])
    print('Extra filter record keys:', list(r['extras_filter'].keys()))
    print('---\n')
