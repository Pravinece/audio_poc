# Round 1 — Audio Pipeline Q&A

---

## Q1. How is the audio captured?

**Browser side — WebRTC**

When user clicks "Start Recording":
```javascript
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
```
- Browser asks user for **mic permission**
- OS gives browser access to the **raw microphone stream**
- This stream is a continuous flow of audio data (PCM samples from mic)

```javascript
mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
mediaRecorder.start(250);
```
- `MediaRecorder` takes that raw stream and **encodes it into webm format** (using Opus codec internally)
- `start(250)` means every 250ms, a chunk of encoded audio is fired via `ondataavailable`

```javascript
mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
```
- Each chunk is pushed into `audioChunks[]` array in **browser RAM**

---

## Q2. How is the audio sent to the backend?

When user clicks "Stop":
```javascript
const blob = new Blob(audioChunks, { type: "audio/webm" });
const formData = new FormData();
formData.append("audio", blob, "recording.webm");
fetch("/evaluate", { method: "POST", body: formData });
```
- All chunks combined into a single **Blob** (still in browser RAM)
- Sent as **HTTP POST multipart/form-data** to Flask `/evaluate`
- Not WebSocket — plain HTTP request

---

## Q3. How does the backend receive the audio?

```python
audio_bytes = request.files["audio"].read()
```
- Flask reads the webm blob as **raw bytes** into Python memory

---

## Q4. Why do we convert webm → wav?

```
Browser records in:   webm (Opus codec) — compressed, small size
PocketSphinx needs:   wav (PCM 16kHz mono) — raw uncompressed samples
librosa needs:        wav (PCM 16kHz mono) — same requirement
```

So we use ffmpeg to convert:
```python
subprocess.run(["ffmpeg", "-i", input.webm, "-ar", "16000", "-ac", "1", output.wav])
```
```
webm (Opus) → ffmpeg decodes → raw PCM samples → saved as wav
```

---

## Q5. Why subprocess for ffmpeg?

FFmpeg is a **standalone C program** — not a Python library. The only way to use it from Python is to spawn it as a separate process via `subprocess.run()`.

This is equivalent to running in terminal:
```
ffmpeg -i input.webm -ar 16000 -ac 1 output.wav
```

---

## Q6. How is the audio used after conversion?

The wav audio is used for **two parallel purposes:**

### Purpose 1 — Transcription (PocketSphinx)
```
wav bytes
    ↓
Strip 44-byte WAV header → raw PCM bytes
    ↓
PocketSphinx internally computes MFCC
(13 numbers per 25ms frame representing vocal tract shape)
    ↓
GMM-HMM model matches MFCC sequence to words
    ↓
Output: "the quick brown fox..."
    ↓
Compare with reference script → WER → Accuracy %
```

### Purpose 2 — Fluency Analysis (librosa)
```
wav bytes
    ↓
librosa.load() → numpy float32 array (amplitude over time)
    ↓
RMS Energy per frame
→ which frames are speech vs silence
→ count speech→silence transitions → pause count
→ speech duration → WPM estimate

pYIN pitch detection
→ F0 (fundamental frequency) per frame
→ std deviation of F0 → pitch variance (intonation stability)
```

---

## Q7. What is MFCC?

**MFCC = Mel Frequency Cepstral Coefficients**

- Represents how **human ears perceive sound** as numbers
- Each frame of audio → 13 coefficients capturing the **vocal tract shape**
- Each phoneme (sound unit) has a unique MFCC fingerprint
- Used internally by PocketSphinx for speech recognition

```
Raw Audio → FFT → Mel Filterbank → Log → DCT → 13 MFCC values per frame
```

---

## Q8. What does librosa do?

librosa is a Python audio signal processing library (CPU only, no GPU needed).

In Round 1 it does 3 things:

| Task | Method | Output |
|---|---|---|
| Load & decode wav | `librosa.load()` | numpy float32 array |
| Voice Activity Detection | `librosa.feature.rms()` | pause count, WPM |
| Pitch detection | `librosa.pyin()` | pitch variance (F0 std) |

---

## Q9. How is the final score computed?

```
Accuracy %     (from PocketSphinx WER)  → 60% weight
Fluency Score  (from librosa features)  → 40% weight

Round 1 Score = Accuracy × 0.6 + Fluency × 0.4
```

Fluency score breakdown:
```
WPM score       → 40%  (ideal range: 120-160 wpm)
Pause score     → 40%  (fewer pauses = better, each pause -10 pts)
Pitch variance  → 20%  (natural range: 20-60 Hz std)
```

---

## Q10. How is memory cleared after evaluation?

| What | Cleared By | When |
|---|---|---|
| `audioChunks[]` | Your code (`audioChunks = []`) | Next recording starts |
| Blob | Browser GC | After `onstop` finishes |
| Mic stream | Your code (`t.stop()`) | On stop button click |
| Temp `.webm` file | Your code (`os.unlink`) | After ffmpeg conversion |
| Temp `.wav` file | Your code (`os.unlink`) | After reading wav bytes |
| `webm_bytes`/`wav_bytes` | Python GC | After request ends |

---

## Q11. How many concurrent users can Round 1 handle?

**Constraint: 8GB RAM / 2 vCPU**

| Server | Concurrent Users |
|---|---|
| Flask dev server | 1 (single threaded) |
| Flask + Gunicorn (2 workers) | 2-4 |
| Flask + Gunicorn + threads | 4-8 |
| FastAPI + Uvicorn | 4-8 |

**Bottleneck:** PocketSphinx ASR takes 3-5 sec per request and is CPU-bound.

```
2 vCPU → 2 parallel decodes max
Each decode = 3-5 sec
= ~24-40 users/minute realistically
```

---

## Q12. Full Audio Flow Diagram

```
Microphone
    ↓
getUserMedia() — browser captures raw PCM from OS
    ↓
MediaRecorder — encodes to webm/Opus, chunks every 250ms
    ↓
audioChunks[] — stored in browser RAM
    ↓
Blob + FormData — combined and sent via HTTP POST
    ↓
Flask /evaluate — receives as bytes
    ↓
ffmpeg subprocess — converts webm → wav 16kHz mono
    ↓
        ┌──────────────────┬─────────────────────┐
        ↓                  ↓                     ↓
  PocketSphinx          librosa RMS          librosa pYIN
  MFCC → GMM-HMM     energy → pauses       F0 → pitch var
  → transcript        → WPM                → variance
        ↓                  └──────────────────────┘
       WER                        fluency score
        ↓                              ↓
    accuracy %                    fluency score
        └──────────────┬───────────────┘
                  Round 1 Score
                  (60/40 weighted)
                       ↓
                  JSON response
                       ↓
                  UI displays results
```

---

## Q13. Why no AI/Deep Learning?

The entire pipeline uses only:
- **GMM-HMM** (PocketSphinx) — statistical acoustic model
- **DSP features** (MFCC, RMS energy, F0) — signal processing
- **Levenshtein distance** (WER) — string comparison algorithm
- **Rule-based weighted formula** — transparent scoring

No neural networks, no pretrained embeddings, no LLMs anywhere.
Every score is **traceable to a measurable feature or rule**.
