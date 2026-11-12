"""Local CAM++ runner and deterministic, non-overlapping ASR frame planning.
Anonymous speaker clusters are scoped to one recording, never real identities.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import wave

POLICY = 'pcm16-mono-16k-midpoint-crossing-overlap-v1'
RESOURCE_DEFAULTS = {
    'DIARIZATION_BIN': 'bin/diarize_demo.real',
    'VAD_MODEL': 'models/fsmnvad-offline_w8a16_a9.adla',
    'SPEAKER_MODEL': 'models/campplus_fixed150_w8a16.adla',
    'VAD_MVN': 'models/am.mvn',
}


def resources():
    root = Path(__file__).resolve().parents[1]
    values = {key: os.environ.get(key) or str(root / relative) for key, relative in RESOURCE_DEFAULTS.items()}
    missing = [key for key, value in values.items() if not value or not Path(value).is_absolute() or not Path(value).is_file()]
    if values['DIARIZATION_BIN'] and not os.access(values['DIARIZATION_BIN'], os.X_OK):
        missing.append('DIARIZATION_BIN (not executable)')
    if missing:
        return None, missing
    fingerprints = {}
    for key, value in values.items():
        h = hashlib.sha256()
        with Path(value).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        fingerprints[key] = h.hexdigest()
    return {'paths': values, 'fingerprints': fingerprints, 'policy': POLICY}, []


def normalize_audio(source, output):
    with wave.open(str(source), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000):
            return source
    executable = shutil.which('ffmpeg')
    if not executable:
        raise ValueError('Speaker mode needs mono 16 kHz PCM16 WAV; install ffmpeg to convert this input, or explicitly use --diarization off')
    target = output / 'normalized.wav'
    try:
        p = subprocess.run([executable, '-nostdin', '-v', 'error', '-y', '-i', str(source), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(target)], capture_output=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Speaker audio normalization timed out') from exc
    if p.returncode:
        raise ValueError('Speaker audio normalization failed: ' + p.stderr.decode(errors='replace')[-500:])
    with wave.open(str(target), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError('Normalized audio format is invalid')
    return target


def run_native(audio, config, timeout):
    paths = config['paths']
    cmd = [paths['DIARIZATION_BIN'], paths['VAD_MODEL'], paths['SPEAKER_MODEL'], paths['VAD_MVN'], str(audio)]
    env = os.environ.copy()
    # Do not inherit the native demo's debugging shortcut to external VAD files.
    env.pop('VAD_SEGMENTS_FILE', None)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Speaker diarization timed out') from exc
    if p.returncode:
        raise ValueError('Speaker diarization failed (exit ' + str(p.returncode) + '): ' + p.stderr[-800:])
    try:
        payload = json.loads(p.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError('Speaker diarization did not return one valid JSON envelope') from exc
    if not isinstance(payload, dict) or payload.get('status') != 'ok':
        raise ValueError('Speaker diarization returned an error response')
    data = payload.get('data')
    if not isinstance(data, dict) or not isinstance(data.get('segments'), list):
        raise ValueError('Speaker diarization response has no segments')
    return data['segments']


def frame_plan(raw, rate, frames, chunk_seconds):
    if not isinstance(raw, list) or not raw or len(raw) > 1000:
        raise ValueError('No speech segments or invalid diarization segment count')
    effective, changes, labels = [], [], {}
    duration = frames / rate
    previous_start = -1
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError('Invalid diarization segment')
        start, end, speaker = item.get('start'), item.get('end'), item.get('speaker')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= duration + .1:
            raise ValueError('Invalid diarization time range')
        if isinstance(speaker, bool) or not isinstance(speaker, int) or not 0 <= speaker < 1000 or start < previous_start:
            raise ValueError('Invalid speaker ID or unordered diarization segments')
        previous_start = start
        if speaker not in labels:
            count = len(labels)
            labels[speaker] = chr(65 + count) if count < 26 else 'S' + str(count + 1)
        current = {'start_frame': min(frames, math.floor(start * rate)), 'end_frame': min(frames, math.ceil(end * rate)), 'speaker_id': speaker, 'speaker_label': labels[speaker], 'raw_indices': [index]}
        if current['start_frame'] >= current['end_frame']:
            raise ValueError('Empty diarization frame range')
        if effective and current['start_frame'] < effective[-1]['end_frame']:
            previous = effective[-1]
            overlap = previous['end_frame'] - current['start_frame']
            if previous['speaker_id'] == speaker:
                previous['end_frame'] = max(previous['end_frame'], current['end_frame'])
                previous['raw_indices'].append(index)
                changes.append({'raw_index': index, 'policy': 'merge_same_speaker', 'overlap_frames': overlap})
                continue
            if overlap > rate or current['end_frame'] <= previous['end_frame']:
                raise ValueError('Ambiguous speaker overlap; cannot safely assign exclusive ASR slices')
            boundary = (previous['end_frame'] + current['start_frame']) // 2
            if not previous['start_frame'] < boundary < current['end_frame']:
                raise ValueError('Invalid speaker overlap boundary')
            previous['end_frame'] = boundary
            current['start_frame'] = boundary
            changes.append({'raw_index': index, 'policy': 'midpoint', 'overlap_frames': overlap, 'boundary_frame': boundary})
        effective.append(current)
    plan = []
    step = max(1, int(chunk_seconds * rate))
    for item in effective:
        for start in range(item['start_frame'], item['end_frame'], step):
            end = min(start + step, item['end_frame'])
            plan.append({'index': len(plan), 'start_frame': start, 'end_frame': end, 'start_seconds': start / rate, 'end_seconds': end / rate, 'speaker_id': item['speaker_id'], 'speaker_label': item['speaker_label']})
    return {'raw_segments': raw, 'effective_segments': effective, 'overlap_adjustments': changes, 'speakers': [{'speaker_id': key, 'label': value} for key, value in labels.items()], 'frame_rate': rate, 'audio_frames': frames, 'plan': plan, 'policy': POLICY}


def validate_prefix(saved, plan):
    if len(saved) > len(plan):
        raise ValueError('ASR checkpoint has too many speaker segments')
    for record, expected in zip(saved, plan):
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in expected.items()) or not isinstance(record.get('text'), str):
            raise ValueError('ASR checkpoint does not match the speaker frame plan')


def timestamp(seconds):
    millis = round(seconds * 1000)
    return f'{millis // 60000:02d}:{millis // 1000 % 60:02d}.{millis % 1000:03d}'


def tagged_text(segments):
    return '\n'.join(f"[{timestamp(s['start_seconds'])}–{timestamp(s['end_seconds'])}] Speaker {s['speaker_label']}: {s['text']}" for s in segments if s['text'])
