"""Distraction Monitor — user-triggered focus mode with VLM-based distraction detection.

Activated by voice command ("focus mode", "help me focus", etc.).
Uses the VLM with multiple camera frames to detect sustained distraction.
Respects user excuses ("I need to check my phone") and pauses during conversation.
"""

import random
import threading
import time
from typing import Optional

from app.config import DistractionConfig
from app.pipeline import play_audio


class DistractionMonitor:
    """Background distraction checker using the VLM and camera ring buffer.

    Only runs when the user explicitly activates focus mode.
    Pauses automatically during conversation so the VLM stays free.
    """

    def __init__(
        self,
        llm,
        camera,
        tts,
        config: DistractionConfig,
        pa_sink: Optional[str] = None,
        console=None,
    ):
        self.llm = llm
        self.camera = camera
        self.tts = tts
        self.config = config
        self.pa_sink = pa_sink
        self.console = console

        self._active = False          # focus mode on/off
        self._paused = False          # paused during conversation
        self._excused_until = 0.0     # timestamp when excuse expires
        self._last_nudge = 0.0        # timestamp of last nudge (for cooldown)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self):
        """Activate focus mode — start background distraction checks."""
        with self._lock:
            if self._active:
                return
            self._active = True
            self._paused = False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._check_loop, daemon=True)
            self._thread.start()
        if self.console:
            self.console.print("  [cyan]Focus mode activated[/cyan]")

    def stop(self):
        """Deactivate focus mode — stop all checks."""
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self.console:
            self.console.print("  [cyan]Focus mode deactivated[/cyan]")

    def pause(self):
        """Pause checks (called when user starts speaking)."""
        self._paused = True

    def resume(self):
        """Resume checks (called after conversation turn ends)."""
        self._paused = False

    def excuse(self, duration: Optional[float] = None):
        """User asked for a break — pause checks for duration seconds."""
        dur = duration or self.config.excuse_duration
        self._excused_until = time.monotonic() + dur
        if self.console:
            self.console.print(f"  [dim]Distraction checks paused for {dur:.0f}s[/dim]")

    def check_text(self, text: str) -> Optional[str]:
        """Check user text for focus mode triggers. Returns action or None.

        Actions: "start", "stop", "excuse"
        """
        lower = text.lower().replace("'", "").replace("\u2019", "")

        for phrase in self.config.start_phrases:
            if phrase in lower:
                return "start"

        for phrase in self.config.stop_phrases:
            if phrase in lower:
                return "stop"

        if self._active:
            for phrase in self.config.excuse_phrases:
                if phrase in lower:
                    return "excuse"

        return None

    def _check_loop(self):
        """Background loop — runs distraction checks at configured interval."""
        while not self._stop_event.is_set():
            # Wait for the interval, checking stop event frequently
            if self._stop_event.wait(timeout=self.config.check_interval):
                break

            # Skip if paused (conversation in progress)
            if self._paused:
                continue

            # Skip if excused
            if time.monotonic() < self._excused_until:
                continue

            # Skip if in cooldown after recent nudge
            if time.monotonic() - self._last_nudge < self.config.cooldown:
                continue

            # Grab frames from ring buffer
            now = time.monotonic()
            frames = self.camera.get_speech_frames(
                speech_start=now - self.config.check_interval,
                speech_end=now,
                max_frames=self.config.frames,
            )

            if not frames:
                continue

            # Skip if paused (re-check after frame grab, user might have started speaking)
            if self._paused or self._stop_event.is_set():
                continue

            # Ask VLM
            result = self._query_vlm(frames)

            if self._paused or self._stop_event.is_set():
                continue

            if result == "DISTRACTED":
                self._nudge()

    def _query_vlm(self, frames: list[str]) -> str:
        """Send frames to VLM with distraction prompt. Returns DISTRACTED or FOCUSED."""
        try:
            full_response = ""
            for chunk_data in self.llm.generate_stream(
                prompt=self.config.prompt,
                images_b64=frames,
                max_tokens=10,
                temperature=0.1,
            ):
                content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
                if content:
                    full_response += content
                # Abort if paused mid-check
                if self._paused or self._stop_event.is_set():
                    return "ABORTED"

            response = full_response.strip().upper()
            if self.console:
                self.console.print(f"  [dim]Distraction check: {response}[/dim]")

            if "DISTRACTED" in response:
                return "DISTRACTED"
            return "FOCUSED"
        except Exception as e:
            if self.console:
                self.console.print(f"  [dim]Distraction check error: {e}[/dim]")
            return "ERROR"

    def _nudge(self):
        """Speak a random nudge message via TTS."""
        self._last_nudge = time.monotonic()
        msg = random.choice(self.config.nudge_messages)

        if self.console:
            self.console.print(f"  [yellow]Nudge:[/yellow] {msg}")

        if self.tts:
            try:
                result = self.tts.synthesize(msg)
                if result.get("audio") is not None:
                    play_audio(result["audio"], result["sample_rate"], sink=self.pa_sink)
            except Exception:
                pass
