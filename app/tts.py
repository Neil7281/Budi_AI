# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# TTS — subprocess-isolated Kokoro TTS.
#
# kokoro-onnx depends on phonemizer-fork (GPL-3.0) and espeak-ng (GPL-3.0).
# To avoid loading GPL code into the same process as NVIDIA CUDA libraries,
# synthesis runs in a separate subprocess (app/tts_worker.py) that
# communicates via JSON lines over stdin/stdout.

import sys
import json
import wave
import base64
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np


VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _download_kokoro_models_if_missing() -> bool:
    """Download Kokoro model and voices to voices/ if not present."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = VOICES_DIR / "kokoro-v1.0.onnx"
    voices_path = VOICES_DIR / "voices-v1.0.bin"
    needed = []
    if not model_path.exists():
        needed.append((KOKORO_MODEL_URL, model_path, "kokoro-v1.0.onnx (~311 MB)"))
    if not voices_path.exists():
        needed.append((KOKORO_VOICES_URL, voices_path, "voices-v1.0.bin (~30 MB)"))
    if not needed:
        return True
    try:
        import httpx
    except ImportError:
        print("Kokoro: install httpx to auto-download models (pip install httpx)")
        return False
    for url, path, label in needed:
        print(f"Downloading {label} to {path} ...")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                done = 0
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=262144):
                        f.write(chunk)
                        done += len(chunk)
                        if total and total > 0:
                            pct = 100 * done / total
                            sys.stdout.write(f"\r  {label}: {pct:.0f}%\r")
                            sys.stdout.flush()
            if total:
                print()
            print(f"  Saved {path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return False
    return True


class KokoroTTS:
    """Kokoro TTS client — synthesis runs in a subprocess for GPL isolation."""

    def __init__(self, voice: str = "af_sarah", speed: float = 1.0, lang: str = "en-us"):
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._proc: Optional[subprocess.Popen] = None
        self._sample_rate = 24000
        self.backend_name = "Kokoro"
        self.provider = "unknown"

    def load(self) -> bool:
        model_path = VOICES_DIR / "kokoro-v1.0.onnx"
        voices_path = VOICES_DIR / "voices-v1.0.bin"
        if not model_path.exists() or not voices_path.exists():
            if not _download_kokoro_models_if_missing():
                return False

        worker = Path(__file__).parent / "tts_worker.py"
        try:
            self._proc = subprocess.Popen(
                [sys.executable, str(worker),
                 "--model-dir", str(VOICES_DIR),
                 "--voice", self.voice,
                 "--speed", str(self.speed),
                 "--lang", self.lang],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # inherit parent's stderr for log visibility
                text=True,
                bufsize=1,
            )
        except Exception as e:
            print(f"TTS worker spawn failed: {e}")
            return False

        line = self._proc.stdout.readline()
        if not line:
            print("TTS worker exited before signalling ready")
            return False

        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            print(f"TTS worker sent invalid init response: {line!r}")
            return False

        if resp.get("status") == "ready":
            self.provider = resp.get("provider", "unknown")
            return True

        print(f"TTS worker error: {resp.get('error', 'unknown')}")
        return False

    def _send(self, req: dict) -> Optional[dict]:
        if not self._proc or self._proc.poll() is not None:
            return None
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                return None
            return json.loads(line)
        except (BrokenPipeError, json.JSONDecodeError, OSError):
            return None

    def synthesize(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {"audio": None, "error": "Empty"}

        resp = self._send({
            "cmd": "synthesize",
            "text": text,
            "voice": self.voice,
            "speed": self.speed,
            "lang": self.lang,
        })
        if resp is None:
            return {"audio": None, "error": "Worker not running"}
        if "error" in resp:
            return {"audio": None, "error": resp["error"]}

        audio_bytes = base64.b64decode(resp["audio_b64"])
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        return {"audio": audio, "sample_rate": resp["sample_rate"]}

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(r["sample_rate"])
            wf.writeframes(r["audio"].tobytes())
        return True

    def health_check(self) -> bool:
        resp = self._send({"cmd": "health"})
        return resp is not None and resp.get("healthy", False)

    def unload(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None


class EdgeTTS:
    """Microsoft Edge TTS — streams audio from Microsoft's cloud service."""

    def __init__(self, voice: str = "en-US-JennyNeural", speed: float = 1.0, lang: str = "en-us"):
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._sample_rate = 24000
        self.backend_name = "Edge"
        self.provider = "cloud"
        self._loaded = False

    def load(self) -> bool:
        try:
            import edge_tts
            self._loaded = True
            return True
        except ImportError:
            print("Edge TTS: install edge-tts (pip install edge-tts)")
            return False

    def synthesize(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {"audio": None, "error": "Empty"}
        if not self._loaded:
            return {"audio": None, "error": "Not loaded"}

        try:
            import asyncio
            import edge_tts
            import io

            rate_str = f"{int((self.speed - 1.0) * 100):+d}%"

            async def _generate():
                communicate = edge_tts.Communicate(text, self.voice, rate=rate_str)
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                return audio_data

            # Run async in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        mp3_data = pool.submit(lambda: asyncio.run(_generate())).result(timeout=30)
                else:
                    mp3_data = loop.run_until_complete(_generate())
            except RuntimeError:
                mp3_data = asyncio.run(_generate())

            if not mp3_data:
                return {"audio": None, "error": "No audio returned"}

            # Decode MP3 to PCM using subprocess (ffmpeg/avconv)
            proc = subprocess.Popen(
                ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ar", "24000",
                 "-ac", "1", "-loglevel", "error", "pipe:1"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            pcm_data, err = proc.communicate(input=mp3_data, timeout=15)

            if proc.returncode != 0 or not pcm_data:
                return {"audio": None, "error": f"ffmpeg decode failed: {err.decode()[:100]}"}

            audio = np.frombuffer(pcm_data, dtype=np.int16)
            return {"audio": audio, "sample_rate": 24000}

        except Exception as e:
            return {"audio": None, "error": str(e)}

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(r["sample_rate"])
            wf.writeframes(r["audio"].tobytes())
        return True

    def health_check(self) -> bool:
        return self._loaded

    def unload(self):
        self._loaded = False


# Edge TTS voice mapping for convenience
EDGE_VOICES = {
    "jenny": "en-US-JennyNeural",
    "aria": "en-US-AriaNeural",
    "guy": "en-US-GuyNeural",
    "sara": "en-US-SaraNeural",
    "andrew": "en-US-AndrewNeural",
    "emma": "en-US-EmmaNeural",
    "brian": "en-US-BrianNeural",
    "sonia": "en-GB-SoniaNeural",
    "ryan": "en-GB-RyanNeural",
}


def create_tts(voice: str = "", speed: float = 1.0, lang: str = "en-us",
               backend: str = "kokoro", **_kwargs):
    """Create the TTS backend.

    backend: "kokoro" (local ONNX) or "edge" (Microsoft Edge cloud TTS)
    """
    if backend == "edge":
        # Resolve short voice names
        resolved = EDGE_VOICES.get(voice.lower(), voice) if voice else "en-US-JennyNeural"
        return EdgeTTS(voice=resolved, speed=speed, lang=lang)
    return KokoroTTS(voice=voice or "af_sarah", speed=speed, lang=lang)
