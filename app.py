import json
import random
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from flask.json.provider import DefaultJSONProvider
from evaluator import evaluate_round1, evaluate_round2, evaluate_round3
from script_bank import SCRIPTS
from word_bank import MEDICAL_WORDS
from qa_bank import QA_BANK


class NumpyJSONProvider(DefaultJSONProvider):
    """Converts numpy types to native Python types before JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
app.secret_key = "r1_eval_secret"


# ---------------------------------------------------------------------------
# Round 1
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    script = random.choice(SCRIPTS)
    session["current_script"] = script
    return render_template("index.html", script=script)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_bytes = request.files["audio"].read()
    reference = request.form.get("reference") or session.get("current_script", "")

    if not reference:
        return jsonify({"error": "No reference script in session"}), 400

    try:
        result = evaluate_round1(audio_bytes, reference)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Round 2
# ---------------------------------------------------------------------------

@app.route("/round2")
def round2():
    word_entry = random.choice(MEDICAL_WORDS)
    session["current_word"] = word_entry
    return render_template("round2.html", word_entry=word_entry)


@app.route("/evaluate_round2", methods=["POST"])
def evaluate_r2():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_bytes = request.files["audio"].read()
    word_entry_raw = request.form.get("word_entry")
    word_entry = json.loads(word_entry_raw) if word_entry_raw else session.get("current_word")

    if not word_entry:
        return jsonify({"error": "No word in session"}), 400

    try:
        result = evaluate_round2(audio_bytes, word_entry)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Round 3
# ---------------------------------------------------------------------------

@app.route("/round3")
def round3():
    qa_entry = random.choice(QA_BANK)
    session["current_qa"] = qa_entry
    return render_template("round3.html", qa_entry=qa_entry)


@app.route("/evaluate_round3", methods=["POST"])
def evaluate_r3():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_bytes = request.files["audio"].read()
    qa_entry_raw = request.form.get("qa_entry")
    qa_entry = json.loads(qa_entry_raw) if qa_entry_raw else session.get("current_qa")

    if not qa_entry:
        return jsonify({"error": "No question in session"}), 400

    try:
        result = evaluate_round3(audio_bytes, qa_entry)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/interview")
def interview():
    return render_template("interview.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
