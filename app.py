import random
from flask import Flask, render_template, request, jsonify, session
from evaluator import evaluate_round1
from script_bank import SCRIPTS

app = Flask(__name__)
app.secret_key = "r1_eval_secret"

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
    reference = session.get("current_script", "")

    if not reference:
        return jsonify({"error": "No reference script in session"}), 400

    try:
        result = evaluate_round1(audio_bytes, reference)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
