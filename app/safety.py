"""Safety Monitor — continuous background VLM-based threat detection.

Runs a dedicated small VLM (on a separate endpoint) to continuously analyze
camera frames for safety threats. When a threat is detected, interrupts any
ongoing TTS playback and speaks an alert message.

Designed to run fully in parallel with the main conversation pipeline:
  - Uses its own LLM instance pointed at a separate model server
  - Reads from the existing camera ring buffer (no extra capture thread)
  - Daemon thread with configurable check interval and cooldown
"""

import json
import random
import subprocess
import threading
import time
from typing import Callable, Optional

from app.config import SafetyConfig
from app.llm import LLM
from app.pipeline import play_audio


class SafetyMonitor:
    """Background safety checker using a dedicated small VLM and camera ring buffer.

    Always-on when enabled. Pauses automatically during conversation turns
    so TTS alerts don't collide with normal responses.
    """

    def __init__(
        self,
        camera,
        tts,
        config: SafetyConfig,
        pa_sink: Optional[str] = None,
        console=None,
        on_alert: Optional[Callable] = None,
    ):
        self.camera = camera
        self.tts = tts
        self.config = config
        self.pa_sink = pa_sink
        self.console = console
        self.on_alert = on_alert  # callback for WebSocket broadcast etc.

        # Dedicated LLM instance for safety — separate endpoint, no contention
        self._llm = LLM(
            model="",
            base_url=config.model_endpoint,
            backend=config.model_backend,
            max_tokens=64,
            temperature=0.1,
            timeout=30.0,
        )

        self._paused = False
        self._last_alert_time = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._tts_interrupt = threading.Event()  # signal to kill active alert playback

        # Consecutive-hit confirmation: require N consecutive THREAT detections
        # before triggering an alert. Eliminates hallucination-driven false positives.
        self._consecutive_threats = 0
        self._confirm_count = config.confirm_count
        self._last_threat_desc = ""

    def load(self) -> bool:
        """Connect to the dedicated safety VLM endpoint. Returns True on success."""
        try:
            ok = self._llm.load()
            if ok and self.console:
                self.console.print(f"  ✓ Safety VLM ({self._llm.model} @ {self.config.model_endpoint})")
            elif not ok and self.console:
                self.console.print(
                    f"  [yellow]⚠ Safety VLM not available at {self.config.model_endpoint} "
                    f"— safety monitor disabled[/yellow]"
                )
            return ok
        except Exception as e:
            if self.console:
                self.console.print(f"  [yellow]⚠ Safety VLM connection failed: {e}[/yellow]")
            return False

    def start(self):
        """Start the background safety monitoring loop."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._check_loop, daemon=True, name="safety-monitor")
            self._thread.start()
        if self.console:
            self.console.print("  [green]✓ Safety monitor active[/green]")

    def stop(self):
        """Stop the background safety monitoring loop."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
            self._tts_interrupt.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self.console:
            self.console.print("  [dim]Safety monitor stopped[/dim]")

    def pause(self):
        """Pause checks (called when user starts speaking / main pipeline active)."""
        self._paused = True
        self._tts_interrupt.set()  # stop any in-progress alert playback

    def resume(self):
        """Resume checks (called after conversation turn ends)."""
        self._paused = False
        self._tts_interrupt.clear()

    def _check_loop(self):
        """Background loop — grabs frames and queries safety VLM at configured interval."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self.config.check_interval):
                break

            # Skip if paused (conversation in progress)
            if self._paused:
                continue

            # Skip if in cooldown
            if time.monotonic() - self._last_alert_time < self.config.cooldown:
                continue

            # Grab latest frame(s) from ring buffer
            now = time.monotonic()
            frames = self.camera.get_speech_frames(
                speech_start=now - self.config.check_interval,
                speech_end=now,
                max_frames=self.config.frames,
            )
            if not frames:
                continue

            # Re-check pause after frame grab
            if self._paused or self._stop_event.is_set():
                continue

            # Query the dedicated safety VLM
            severity, description = self._query_safety_vlm(frames)

            if self._paused or self._stop_event.is_set():
                continue

            if severity in ("WARNING", "CRITICAL"):
                self._consecutive_threats += 1
                self._last_threat_desc = description
                if self.console:
                    self.console.print(
                        f"  [dim]Safety: threat {self._consecutive_threats}/{self._confirm_count}[/dim]"
                    )
                if self._consecutive_threats >= self._confirm_count:
                    self._trigger_alert(severity, description)
                    self._consecutive_threats = 0
            else:
                # Reset streak on any SAFE response
                self._consecutive_threats = 0

    def _vlm_ask(self, prompt: str, frames: list[str], max_tokens: int = 10) -> str:
        """Send a single prompt+frames to the safety VLM. Returns stripped response."""
        full = ""
        for chunk_data in self._llm.generate_stream(
            prompt=prompt,
            images_b64=frames,
            max_tokens=max_tokens,
            temperature=0.1,
        ):
            content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
            if content:
                full += content
            if self._paused or self._stop_event.is_set():
                return ""
        return full.strip()

    def _query_safety_vlm(self, frames: list[str]) -> tuple[str, str]:
        """Two-step safety check: first YES/NO, then describe if YES.

        Returns (severity, description).
        severity is one of: "SAFE", "WARNING", "ERROR"
        """
        try:
            # Step 1: Simple YES/NO gate — hard for small models to hallucinate
            response = self._vlm_ask(self.config.prompt, frames, max_tokens=5)

            if self.console:
                self.console.print(f"  [dim]Safety check: {response}[/dim]")

            if not response:
                return ("ABORTED", "")

            # Only "YES" (possibly with punctuation) counts as a threat
            first_word = response.upper().split()[0].strip(".,!") if response.split() else ""
            if first_word != "YES":
                return ("SAFE", "")

            # Step 2: Ask what the danger is (only reached on confirmed YES)
            if self._paused or self._stop_event.is_set():
                return ("ABORTED", "")

            desc_response = self._vlm_ask(
                "What is the danger you see? Answer in one short sentence.",
                frames,
                max_tokens=32,
            )

            description = desc_response.strip() if desc_response else "potential danger detected"
            if self.console:
                self.console.print(f"  [dim]Safety detail: {description}[/dim]")

            return ("WARNING", description)

        except Exception as e:
            if self.console:
                self.console.print(f"  [dim]Safety check error: {e}[/dim]")
            return ("ERROR", "")

    def _trigger_alert(self, severity: str, description: str):
        """Interrupt current audio and speak a safety alert via TTS."""
        self._last_alert_time = time.monotonic()
        self._tts_interrupt.clear()

        # Pick a random alert message and fill in the description
        template = random.choice(self.config.alert_messages)
        message = template.format(description=description)

        if self.console:
            color = "red" if severity == "CRITICAL" else "yellow"
            self.console.print(f"  [{color}]⚠ SAFETY {severity}:[/{color}] {description}")

        # Notify external listeners (WebSocket broadcast, etc.)
        if self.on_alert:
            try:
                self.on_alert({
                    "type": "safety_alert",
                    "severity": severity.lower(),
                    "description": description,
                    "message": message,
                    "timestamp": time.time(),
                })
            except Exception:
                pass

        # Kill any currently playing audio (paplay/aplay) to interrupt main TTS
        self._kill_audio_playback()

        # Speak the alert
        if self.tts and not self._tts_interrupt.is_set():
            try:
                result = self.tts.synthesize(f"{self.config.alert_prefix} {message}")
                if result.get("audio") is not None and not self._tts_interrupt.is_set():
                    play_audio(result["audio"], result["sample_rate"], sink=self.pa_sink)
            except Exception as e:
                if self.console:
                    self.console.print(f"  [dim]Safety TTS error: {e}[/dim]")

    @staticmethod
    def _kill_audio_playback():
        """Kill any active paplay/aplay processes to interrupt current TTS output."""
        for proc_name in ("paplay", "aplay"):
            try:
                subprocess.run(
                    ["pkill", "-f", proc_name],
                    capture_output=True, timeout=2,
                )
            except Exception:
                pass
