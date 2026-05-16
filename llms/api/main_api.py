import os
import sys
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, make_response, request


LLMS_DIR = Path(__file__).resolve().parents[1]
if str(LLMS_DIR) not in sys.path:
    sys.path.insert(0, str(LLMS_DIR))

from inferencia import LLMInference  # noqa: E402



DEFAULT_CONFIG_PATH = LLMS_DIR / "./config.json"
DEFAULT_WEIGHTS_PATH = LLMS_DIR / "./model_weights.pth"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
CORS_HEADERS = "Content-Type, Authorization"
CORS_METHODS = "GET, POST, OPTIONS"


def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = CORS_ORIGINS
    response.headers["Access-Control-Allow-Headers"] = CORS_HEADERS
    response.headers["Access-Control-Allow-Methods"] = CORS_METHODS
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


def _to_int(value, default, field_name, minimum=None):
    if value is None:
        return default

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} precisa ser um inteiro.") from exc

    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} precisa ser >= {minimum}.")

    return parsed


def _to_float(value, default, field_name, minimum=None):
    if value is None:
        return default

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} precisa ser um numero.") from exc

    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} precisa ser >= {minimum}.")

    return parsed


def create_app(
    config_path=None,
    weights_path=None,
    device=None,
):
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            return _add_cors_headers(make_response("", 204))

    @app.after_request
    def add_cors_headers(response):
        return _add_cors_headers(response)

    resolved_config_path = Path(
        config_path or os.getenv("LLM_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    )
    resolved_weights_path = Path(
        weights_path or os.getenv("LLM_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH)
    )
    resolved_device = device or os.getenv("LLM_DEVICE", "auto")

    inference = LLMInference(
        config_path=resolved_config_path,
        weights_path=resolved_weights_path,
        device=resolved_device,
    )
    generation_lock = Lock()

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "llm-idl-api",
                "status": "ok",
                "routes": {
                    "health": "GET /health",
                    "config": "GET /config",
                    "generate": "POST /generate",
                },
            }
        )

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "device": str(inference.device),
                "config_path": str(resolved_config_path),
                "weights_path": str(resolved_weights_path),
            }
        )

    @app.get("/config")
    def model_config():
        return jsonify(
            {
                "model_config": inference.config.get("model_config", {}),
                "training_config": inference.config.get("training_config", {}),
                "dataset_info": inference.config.get("dataset_info", {}),
                "metadata": inference.config.get("metadata", {}),
                "device": str(inference.device),
            }
        )

    @app.route("/generate", methods=["POST", "OPTIONS"])
    def generate():
        if request.method == "OPTIONS":
            return _add_cors_headers(make_response("", 204))

        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"error": "Campo 'prompt' e obrigatorio e deve ser texto."}), 400

        try:
            max_new_tokens = _to_int(
                data.get("max_token", data.get("max_new_tokens")),
                default=80,
                field_name="max_token",
                minimum=1,
            )
            temperature = _to_float(
                data.get("temperature"),
                default=0.9,
                field_name="temperature",
                minimum=0.0001,
            )
            top_k = _to_int(
                data.get("top_k"),
                default=None,
                field_name="top_k",
                minimum=1,
            )
            top_p = _to_float(
                data.get("top_p"),
                default=None,
                field_name="top_p",
                minimum=0.0001,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if top_p is not None and top_p > 1:
            return jsonify({"error": "top_p precisa ser <= 1."}), 400

        with generation_lock:
            text = inference.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        return jsonify(
            {
                "prompt": prompt,
                "response": text,
                "params": {
                    "max_token": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                },
                "device": str(inference.device),
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"

    app.run(host=host, port=port, debug=debug)
