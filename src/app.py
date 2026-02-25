from flask import Flask, request, jsonify
from rag_pipeline import answer_query

app = Flask(__name__)


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"]

    try:
        result = answer_query(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def health():
    return jsonify({"status": "RAG API is running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
