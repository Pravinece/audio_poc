import io
import os
import tempfile
import subprocess
import numpy as np
import librosa
from jiwer import wer as compute_jiwer_wer
from pocketsphinx import Decoder, get_model_path


# ---------------------------------------------------------------------------
# Audio conversion: webm (browser WebRTC) -> wav 16kHz mono (PocketSphinx)
# ---------------------------------------------------------------------------

def _webm_to_wav_bytes(webm_bytes: bytes) -> bytes:
    """Convert webm audio bytes to wav 16kHz mono using ffmpeg subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src:
        src.write(webm_bytes)
        src_path = src.name

    dst_path = src_path.replace(".webm", ".wav")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                dst_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        with open(dst_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(src_path)
        if os.path.exists(dst_path):
            os.unlink(dst_path)


# ---------------------------------------------------------------------------
# ASR: PocketSphinx GMM-HMM transcription
# ---------------------------------------------------------------------------

def transcribe_audio(wav_bytes: bytes) -> str:
    """Transcribe 16kHz mono WAV bytes using PocketSphinx."""
    config = Decoder.default_config()
    config.set_string("-hmm",  get_model_path("en-us"))
    config.set_string("-lm",   get_model_path("en-us.lm.bin"))
    config.set_string("-dict", get_model_path("cmudict-en-us.dict"))
    config.set_string("-logfn", os.devnull)   # suppress PocketSphinx logs

    decoder = Decoder(config)

    # Strip 44-byte WAV header to get raw PCM
    raw_pcm = wav_bytes[44:]

    decoder.start_utt()
    decoder.process_raw(raw_pcm, no_search=False, full_utt=True)
    decoder.end_utt()

    hyp = decoder.hyp()
    return hyp.hypstr.lower().strip() if hyp else ""


# ---------------------------------------------------------------------------
# WER & misread words
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation."""
    import re
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    if not hypothesis:
        return 1.0
    return compute_jiwer_wer(_normalize(reference), _normalize(hypothesis))


def get_misread_words(reference: str, hypothesis: str) -> list:
    ref_words = _normalize(reference).split()
    hyp_words = set(_normalize(hypothesis).split())
    return [w for w in ref_words if w not in hyp_words]


# ---------------------------------------------------------------------------
# Fluency features via librosa
# ---------------------------------------------------------------------------

def compute_fluency(wav_bytes: bytes) -> dict:
    """Extract speech rate, pause count, pitch variance from WAV bytes."""
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=16000, mono=True)

    duration_sec = librosa.get_duration(y=y, sr=sr)

    # VAD via RMS energy
    hop_length = 512
    energy = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop_length)[0]
    threshold = np.mean(energy) * 0.3
    is_speech = energy > threshold

    # Pause count = number of speech->silence transitions
    pause_count = int(np.sum(np.diff(is_speech.astype(int)) == -1))

    # Pitch variance (F0 std dev)
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=300, sr=sr)
    f0_valid = f0[~np.isnan(f0)]
    pitch_variance = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0

    # Approximate WPM: speech duration * (4 syllables/sec / 2.5 syllables/word) * 60
    speech_duration = (np.sum(is_speech) * hop_length) / sr
    approx_wpm = round(
        (speech_duration * 4 / 2.5) * (60 / max(duration_sec, 1)), 1
    )

    return {
        "duration_sec":    round(duration_sec, 2),
        "pause_count":     pause_count,
        "pitch_variance":  round(pitch_variance, 2),
        "approx_wpm":      approx_wpm,
    }


def compute_fluency_score(fluency: dict) -> float:
    """
    Score 0-100:
      WPM         40%  — ideal 120-160
      Pause count 40%  — fewer is better (each pause -10 pts)
      Pitch var   20%  — natural range 20-60 Hz std
    """
    wpm = fluency["approx_wpm"]
    if 120 <= wpm <= 160:
        wpm_score = 100.0
    elif wpm < 120:
        wpm_score = max(0.0, (wpm / 120) * 100)
    else:
        wpm_score = max(0.0, 100 - ((wpm - 160) / 40) * 20)

    pause_score = max(0.0, 100 - fluency["pause_count"] * 10)

    pv = fluency["pitch_variance"]
    if 20 <= pv <= 60:
        pitch_score = 100.0
    elif pv < 20:
        pitch_score = max(0.0, (pv / 20) * 100)
    else:
        pitch_score = max(0.0, 100 - ((pv - 60) / 40) * 30)

    return round(wpm_score * 0.4 + pause_score * 0.4 + pitch_score * 0.2, 2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_round1(webm_bytes: bytes, reference_script: str) -> dict:
    """
    Full Round 1 pipeline:
      webm -> wav -> ASR -> WER -> fluency -> score
    Final score = 60% accuracy + 40% fluency
    """
    wav_bytes = _webm_to_wav_bytes(webm_bytes)

    transcript    = transcribe_audio(wav_bytes)
    error_rate    = compute_wer(reference_script, transcript)
    accuracy      = round((1 - error_rate) * 100, 2)
    misread       = get_misread_words(reference_script, transcript)
    fluency       = compute_fluency(wav_bytes)
    fluency_score = compute_fluency_score(fluency)
    round1_score  = round(accuracy * 0.6 + fluency_score * 0.4, 2)

    return {
        "transcript":       transcript,
        "reference":        reference_script,
        "accuracy_percent": accuracy,
        "wer":              round(error_rate, 4),
        "misread_words":    misread,
        "fluency":          fluency,
        "fluency_score":    fluency_score,
        "round1_score":     round1_score,
    }
