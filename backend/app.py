"""
app.py
Flask backend - Pure Python NLP, no API keys needed.
"""

import os
import sys
import io

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from services.file_parser import extract_text, chunk_text
from services.ai_analyzer import analyze_document
from services.pdf_exporter import generate_report_pdf
from services.doc_corrector import correct_docx

# Serve frontend from ../frontend folder
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Max upload size: 20MB
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "docx"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    """Serve the frontend."""
    return app.send_static_file("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Upload and analyze a document."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload PDF or DOCX."}), 415

    try:
        file_bytes = file.read()

        # Extract text
        text = extract_text(file.filename, file_bytes)

        if not text.strip():
            return jsonify({"error": "Could not extract text. File may be scanned or image-based."}), 422

        # Chunk for large documents
        chunks = chunk_text(text)

        # AI Analysis (pure Python NLP)
        analysis = analyze_document(text, chunks)

        return jsonify({
            "filename": file.filename,
            "word_count": len(text.split()),
            "char_count": len(text),
            "chunk_count": len(chunks),
            "extracted_text": text[:5000],
            "full_text_available": len(text) > 5000,
            **analysis
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 415
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/download-report", methods=["POST"])
def download_report():
    """Generate and download a PDF report."""
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        pdf_bytes = generate_report_pdf(
            filename=data.get("filename", "document"),
            analysis=data
        )
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="analysis_report.pdf"
        )
    except Exception as e:
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500


@app.route("/api/download-corrected", methods=["POST"])
def download_corrected():
    """Generate and download a spelling-corrected DOCX."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith(".docx"):
        return jsonify({"error": "Only DOCX files can be corrected currently."}), 415

    try:
        file_bytes = file.read()
        corrected_bytes = correct_docx(file_bytes)
        
        return send_file(
            io.BytesIO(corrected_bytes),
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True,
            download_name=f"Corrected_{file.filename}"
        )
    except Exception as e:
        return jsonify({"error": f"Correction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*60)
    print("  AI DOCUMENT ANALYZER - Pure Python NLP")
    print("  No API keys needed - 100% local processing")
    print("="*60)
    print(f"\n  Server running at: http://localhost:{port}")
    print("  Open this URL in your browser to use the app\n")
    app.run(debug=False, host="0.0.0.0", port=port)
