from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable


BASE_URL = "https://api.elevenlabs.io/v1"


def get_requests():
    try:
        import requests  # type: ignore
    except ImportError:
        raise SystemExit(
            "Missing dependency: requests. Install with `python -m pip install requests`."
        )
    return requests


def load_dotenv(path: str = ".env") -> None:
    """Lightweight .env loader to keep secrets project-local without extra deps."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("Set ELEVENLABS_API_KEY in .env (project-local)")
    return api_key


def request_json_or_text(response) -> str:
    try:
        return json.dumps(response.json(), indent=2)
    except ValueError:
        return response.text


def list_voices(api_key: str) -> list[dict[str, Any]]:
    requests = get_requests()
    response = requests.get(
        f"{BASE_URL}/voices",
        headers={"xi-api-key": api_key, "Accept": "application/json"},
        timeout=30,
    )
    if not response.ok:
        raise SystemExit(
            f"List voices failed ({response.status_code}): {request_json_or_text(response)}"
        )
    payload = response.json()
    return payload.get("voices", [])


def print_voices(voices: Iterable[dict[str, Any]]) -> None:
    for voice in voices:
        name = voice.get("name") or ""
        voice_id = voice.get("voice_id") or ""
        category = voice.get("category") or ""
        label = f"{name} [{category}]" if category else name
        print(f"{label} - {voice_id}")


def pick_voice_by_name(
    voices: Iterable[dict[str, Any]], name_query: str
) -> dict[str, Any]:
    query = name_query.strip().lower()
    matches = [
        voice
        for voice in voices
        if query in (voice.get("name") or "").lower()
    ]
    if not matches:
        raise SystemExit(f"No voice matched name query: {name_query!r}")
    if len(matches) > 1:
        print("Multiple voices matched. Be more specific or use --voice-id.")
        print_voices(matches)
        raise SystemExit(1)
    return matches[0]


def resolve_voice_id(api_key: str, args: argparse.Namespace) -> str:
    if args.voice_id:
        return args.voice_id

    env_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if env_voice_id:
        return env_voice_id

    voices = list_voices(api_key)

    if args.voice_name:
        return pick_voice_by_name(voices, args.voice_name)["voice_id"]

    if not voices:
        raise SystemExit("No voices returned. Check your ElevenLabs account.")

    first = voices[0]
    voice_id = first.get("voice_id")
    name = first.get("name") or "unknown"
    if not voice_id:
        raise SystemExit("First voice is missing a voice_id.")
    print(f"Using first available voice: {name} ({voice_id})")
    return voice_id


def resolve_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("Provide --text, --text-file, or pipe text via stdin.")


def synthesize_bytes(
    api_key: str,
    voice_id: str,
    text: str,
    model_id: str | None,
    output_format: str | None,
    enable_logging: bool,
    voice_settings: dict[str, Any] | None,
) -> bytes:
    requests = get_requests()
    params: dict[str, str] = {}
    if output_format:
        params["output_format"] = output_format
    if not enable_logging:
        params["enable_logging"] = "false"

    payload: dict[str, Any] = {"text": text}
    if model_id:
        payload["model_id"] = model_id
    if voice_settings:
        payload["voice_settings"] = voice_settings

    response = requests.post(
        f"{BASE_URL}/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "*/*",
        },
        params=params,
        json=payload,
        timeout=120,
    )
    if not response.ok:
        raise SystemExit(
            f"TTS request failed ({response.status_code}): {request_json_or_text(response)}"
        )

    return response.content


def synthesize(
    api_key: str,
    voice_id: str,
    text: str,
    output_path: str,
    model_id: str | None,
    output_format: str | None,
    enable_logging: bool,
    voice_settings: dict[str, Any] | None,
) -> None:
    audio = synthesize_bytes(
        api_key=api_key,
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        output_format=output_format,
        enable_logging=enable_logging,
        voice_settings=voice_settings,
    )
    Path(output_path).write_bytes(audio)
    print(f"Saved audio to {output_path}")


def play_audio(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["afplay", path], check=False)
        return
    if sys.platform.startswith("linux"):
        for player in (["paplay", path], ["aplay", path]):
            try:
                subprocess.run(player, check=False)
                return
            except FileNotFoundError:
                continue
    if sys.platform.startswith("win"):
        subprocess.run(["cmd", "/c", "start", "", path], check=False)
        return
    print("Playback not supported on this platform.")


def load_voice_map() -> dict[str, str]:
    raw = os.getenv("ELEVENLABS_VOICE_MAP")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid ELEVENLABS_VOICE_MAP JSON: {exc}")
    if not isinstance(payload, dict):
        raise SystemExit("ELEVENLABS_VOICE_MAP must be a JSON object")
    voice_map: dict[str, str] = {}
    for key, value in payload.items():
        if key is None or value is None:
            continue
        voice_map[str(key).strip().lower()] = str(value).strip()
    return voice_map


class ServerConfig:
    def __init__(
        self,
        api_key: str,
        default_voice_id: str | None,
        voice_map: dict[str, str],
        output_format: str | None,
        model_id: str | None,
        enable_logging: bool,
        cors_origin: str,
    ) -> None:
        self.api_key = api_key
        self.default_voice_id = default_voice_id
        self.voice_map = voice_map
        self.output_format = output_format
        self.model_id = model_id
        self.enable_logging = enable_logging
        self.cors_origin = cors_origin


class TTSServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], config: ServerConfig) -> None:
        super().__init__(server_address, TTSRequestHandler)
        self.config = config


class TTSRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self.server.config.cors_origin)
        self.end_headers()
        self.wfile.write(body)

    def _send_audio(self, audio: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Access-Control-Allow-Origin", self.server.config.cors_origin)
        self.end_headers()
        self.wfile.write(audio)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.server.config.cors_origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/tts":
            self._send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        text = str(payload.get("text") or "").strip()
        if not text:
            self._send_json(400, {"error": "Missing text"})
            return

        name = str(payload.get("name") or payload.get("speaker") or "").strip()
        voice_id = str(payload.get("voice_id") or "").strip()
        if not voice_id:
            key = name.lower()
            voice_id = self.server.config.voice_map.get(key, "")
        if not voice_id:
            voice_id = self.server.config.default_voice_id or ""
        if not voice_id:
            self._send_json(
                400,
                {
                    "error": "No voice_id resolved. Provide voice_id, "
                    "set ELEVENLABS_VOICE_ID, or map name via ELEVENLABS_VOICE_MAP."
                },
            )
            return

        model_id = payload.get("model_id") or self.server.config.model_id
        output_format = payload.get("output_format") or self.server.config.output_format
        voice_settings = payload.get("voice_settings")
        if voice_settings is not None and not isinstance(voice_settings, dict):
            self._send_json(400, {"error": "voice_settings must be an object"})
            return

        try:
            audio = synthesize_bytes(
                api_key=self.server.config.api_key,
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                output_format=output_format,
                enable_logging=self.server.config.enable_logging,
                voice_settings=voice_settings,
            )
        except SystemExit as exc:
            self._send_json(502, {"error": str(exc)})
            return

        self._send_audio(audio)


def serve(api_key: str, args: argparse.Namespace) -> None:
    voice_map = load_voice_map()
    default_voice_id = None
    if args.voice_id or args.voice_name or os.getenv("ELEVENLABS_VOICE_ID"):
        default_voice_id = resolve_voice_id(api_key, args)

    config = ServerConfig(
        api_key=api_key,
        default_voice_id=default_voice_id,
        voice_map=voice_map,
        output_format=args.output_format,
        model_id=args.model_id,
        enable_logging=not args.no_logging,
        cors_origin=args.cors_origin,
    )
    server = TTSServer((args.host, args.port), config)
    print(f"TTS server listening on http://{args.host}:{args.port}/tts")
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ElevenLabs text-to-speech test")
    parser.add_argument("--list-voices", action="store_true", help="List voices and exit")
    parser.add_argument("--serve", action="store_true", help="Run HTTP server for TTS")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--cors-origin",
        default="*",
        help="CORS origin for server responses (default: *)",
    )
    parser.add_argument("--voice-id", help="Voice ID to use")
    parser.add_argument("--voice-name", help="Voice name (partial match)")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--text-file", help="Path to a text file to synthesize")
    parser.add_argument("--model-id", help="Model ID (optional)")
    parser.add_argument(
        "--output-format",
        default="mp3_44100_128",
        help="Audio output format (default: mp3_44100_128)",
    )
    parser.add_argument("--output", default="out.mp3", help="Output audio file path")
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable request logging (if allowed by your plan)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip automatic playback after synthesis",
    )
    parser.add_argument(
        "--voice-settings",
        help='JSON for voice_settings, e.g. \'{"stability":0.5,"similarity_boost":0.8}\'',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = get_api_key()

    if args.list_voices:
        voices = list_voices(api_key)
        print_voices(voices)
        return
    if args.serve:
        serve(api_key, args)
        return

    voice_id = resolve_voice_id(api_key, args)
    text = resolve_text(args)
    voice_settings = None
    if args.voice_settings:
        try:
            voice_settings = json.loads(args.voice_settings)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --voice-settings JSON: {exc}")

    synthesize(
        api_key=api_key,
        voice_id=voice_id,
        text=text,
        output_path=args.output,
        model_id=args.model_id,
        output_format=args.output_format,
        enable_logging=not args.no_logging,
        voice_settings=voice_settings,
    )
    if not args.no_play:
        play_audio(args.output)


if __name__ == "__main__":
    main()
