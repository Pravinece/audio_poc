import io
import os
import re
import json
import tempfile
import subprocess
import numpy as np
import librosa
from g2p_en import G2p
from vosk import Model, KaldiRecognizer
from jiwer import wer as compute_jiwer_wer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz

# Load models once at module level
_VOSK_MODEL  = Model(model_path="models/vosk-model-small-en-us-0.15")
_G2P         = G2p()
_LEMMATIZER  = WordNetLemmatizer()
_STOPWORDS   = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Audio conversion: webm (browser WebRTC) -> wav 16kHz mono (PocketSphinx)
# ---------------------------------------------------------------------------

FFMPEG_PATH = (
    r"C:\Users\p1569\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
)


def _webm_to_wav_bytes(webm_bytes: bytes) -> bytes:
    """Convert webm audio bytes to wav 16kHz mono using ffmpeg subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src:
        src.write(webm_bytes)
        src_path = src.name

    dst_path = src_path.replace(".webm", ".wav")

    try:
        subprocess.run(
            [
                FFMPEG_PATH, "-y",
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
# ASR: Vosk transcription
# ---------------------------------------------------------------------------

def transcribe_audio(wav_bytes: bytes) -> str:
    """Transcribe 16kHz mono WAV bytes using Vosk."""
    recognizer = KaldiRecognizer(_VOSK_MODEL, 16000)
    recognizer.SetWords(True)

    # Feed raw PCM (skip 44-byte WAV header)
    raw_pcm = wav_bytes[44:]

    # Feed in chunks of 4000 bytes
    chunk_size = 4000
    for i in range(0, len(raw_pcm), chunk_size):
        recognizer.AcceptWaveform(raw_pcm[i:i + chunk_size])

    result = json.loads(recognizer.FinalResult())
    return result.get("text", "").lower().strip()


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
# Round 2 — Pronunciation Evaluation
# ---------------------------------------------------------------------------

def _get_candidate_phonemes(transcript: str, reference_word: str) -> list:
    """
    Convert transcribed word to CMU phonemes using g2p_en.
    Strips stress markers (e.g. EH1 -> EH) to match word_bank format.
    Falls back to reference word if transcript is empty.
    """
    word = transcript.strip().lower().split()[0] if transcript.strip() else reference_word.lower()
    raw  = _G2P(word)
    # strip stress numbers and spaces/punctuation tokens
    return [re.sub(r"\d", "", p) for p in raw if p.isalpha() or p[:-1].isalpha()]


def _align_phonemes(ref: list, hyp: list) -> dict:
    """
    Align reference and hypothesis phoneme sequences using
    Levenshtein edit distance to find matched/substituted/deleted phonemes.
    Returns per-phoneme comparison result.
    """
    n, m = len(ref), len(hyp)
    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrack to find alignment
    i, j = n, m
    correct, wrong = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            correct.append(ref[i-1])
            i -= 1; j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] <= dp[i-1][j]):
            j -= 1  # insertion in hyp
        else:
            wrong.append(ref[i-1])  # deletion or substitution
            i -= 1

    return {"correct": correct, "wrong": wrong}


def _compute_mfcc_score(wav_bytes: bytes, reference_word: str) -> float:
    """
    Compute MFCC-based pronunciation score.
    Measures how stable and clear the acoustic features are.
    Higher spectral clarity = better articulation.
    """
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=16000, mono=True)

    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) < 1000:
        return 0.0

    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)

    # Score based on MFCC stability (low variance across frames = clear articulation)
    mfcc_std = np.mean(np.std(mfcc, axis=1))

    # Normalize: typical good pronunciation has std between 10-30
    if mfcc_std <= 30:
        score = 100.0
    else:
        score = max(0.0, 100 - ((mfcc_std - 30) / 20) * 30)

    return round(score, 2)


def evaluate_round2(webm_bytes: bytes, word_entry: dict) -> dict:
    """
    Full Round 2 pipeline:
      webm -> wav -> ASR -> phoneme lookup -> alignment -> GOP score
    Final score = phoneme_accuracy×0.5 + word_match×0.3 + mfcc_score×0.2
    """
    wav_bytes     = _webm_to_wav_bytes(webm_bytes)
    transcript    = transcribe_audio(wav_bytes)

    reference_word    = word_entry["word"]
    reference_phonemes = word_entry["phonemes"]

    # Get candidate phonemes from transcript
    candidate_phonemes = _get_candidate_phonemes(transcript, reference_word)

    # Align and compare phoneme sequences
    alignment = _align_phonemes(reference_phonemes, candidate_phonemes)
    total     = len(reference_phonemes)
    matched   = len(alignment["correct"])
    phoneme_accuracy = round((matched / total) * 100, 2) if total > 0 else 0.0

    # Word match score: did candidate say the exact word?
    word_match = 1.0 if reference_word.lower() in transcript.lower() else 0.0

    # MFCC acoustic score
    mfcc_score = _compute_mfcc_score(wav_bytes, reference_word)

    # GOP approximation
    gop_score = round(
        phoneme_accuracy * 0.5 + word_match * 100 * 0.3 + mfcc_score * 0.2, 2
    )

    return {
        "transcript":          transcript,
        "reference_word":      reference_word,
        "ipa":                 word_entry["ipa"],
        "reference_phonemes":  reference_phonemes,
        "candidate_phonemes":  candidate_phonemes,
        "correct_phonemes":    alignment["correct"],
        "wrong_phonemes":      alignment["wrong"],
        "phoneme_accuracy":    phoneme_accuracy,
        "word_match":          bool(word_match),
        "mfcc_score":          mfcc_score,
        "round2_score":        gop_score,
    }


# ---------------------------------------------------------------------------
# Round 3 — Q&A Speaking Evaluation
# ---------------------------------------------------------------------------

def _preprocess(text: str) -> str:
    """Lowercase, remove punctuation, remove stopwords, lemmatize."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = [
        _LEMMATIZER.lemmatize(w)
        for w in text.split()
        if w not in _STOPWORDS
    ]
    return " ".join(tokens)


def _expand_synonyms(text: str) -> str:
    """Expand text with WordNet synonyms to catch paraphrases."""
    expanded = set(text.split())
    for word in text.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return " ".join(expanded)


def _tfidf_similarity(reference: str, candidate: str) -> float:
    """Compute TF-IDF cosine similarity between reference and candidate."""
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([reference, candidate])
        score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return round(float(score) * 100, 2)
    except Exception:
        return 0.0


def _fuzzy_score(reference: str, candidate: str) -> float:
    """RapidFuzz partial ratio to tolerate ASR transcription errors."""
    return round(fuzz.partial_ratio(reference, candidate), 2)


def _keyword_score(candidate: str, keywords: list) -> float:
    """Percentage of reference keywords present in candidate answer."""
    candidate_words = set(_normalize(candidate).split())
    matched = [k for k in keywords if k.lower() in candidate_words]
    return round((len(matched) / len(keywords)) * 100, 2) if keywords else 0.0


def _classify_answer(score: float) -> str:
    if score >= 70: return "correct"
    if score >= 40: return "partially correct"
    return "incorrect"


def evaluate_round3(webm_bytes: bytes, qa_entry: dict) -> dict:
    """
    Full Round 3 pipeline:
      webm -> wav -> ASR -> NLP matching -> score
    Final score = tfidf×0.5 + fuzzy×0.2 + keyword×0.3
    """
    wav_bytes  = _webm_to_wav_bytes(webm_bytes)
    transcript = transcribe_audio(wav_bytes)

    reference  = qa_entry["reference_answer"]
    keywords   = qa_entry["keywords"]

    # Preprocess both texts
    ref_clean  = _preprocess(reference)
    cand_clean = _preprocess(transcript)

    # Expand with synonyms for paraphrase tolerance
    ref_expanded  = _expand_synonyms(ref_clean)
    cand_expanded = _expand_synonyms(cand_clean)

    # Three scoring signals
    tfidf_score   = _tfidf_similarity(ref_expanded, cand_expanded)
    fuzzy_score   = _fuzzy_score(ref_clean, cand_clean)
    keyword_score = _keyword_score(transcript, keywords)

    # Matched and missing keywords
    candidate_words  = set(_normalize(transcript).split())
    matched_keywords = [k for k in keywords if k.lower() in candidate_words]
    missing_keywords = [k for k in keywords if k.lower() not in candidate_words]

    # Final weighted score
    round3_score = round(
        tfidf_score * 0.5 + fuzzy_score * 0.2 + keyword_score * 0.3, 2
    )
    verdict = _classify_answer(round3_score)

    return {
        "transcript":       transcript,
        "question":         qa_entry["question"],
        "reference_answer": reference,
        "tfidf_score":      tfidf_score,
        "fuzzy_score":      fuzzy_score,
        "keyword_score":    keyword_score,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "verdict":          verdict,
        "round3_score":     round3_score,
    }


# ---------------------------------------------------------------------------
# Main entry point Round 1
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
