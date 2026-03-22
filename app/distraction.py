"""Distraction Monitor — user-triggered focus mode with VLM-based distraction detection.

Activated by voice command ("focus mode", "help me focus", etc.).
Uses the VLM with multiple camera frames to detect sustained distraction.
Respects user excuses ("I need to check my phone") and pauses during conversation.
"""

import base64
import re
import random
import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.config import DistractionConfig
from app.pipeline import play_audio

# Words in VLM response that indicate it granted a pause/excuse
_GRANT_PATTERNS = re.compile(
    r"\b(go ahead|take your time|no problem|of course|sure thing|"
    r"ill wait|i will wait|ill pause|i will pause|"
    r"take a break|no worries|alright|okay go|"
    r"be quick|hurry back|come back|ill be here|i will be here)\b",
    re.I,
)

# Smaller resolution for distraction checks — reduces token count significantly
_DISTRACTION_WIDTH = 320
_DISTRACTION_HEIGHT = 240
_DISTRACTION_JPEG_QUALITY = 50


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
        on_nudge=None,
    ):
        self.llm = llm
        self.camera = camera
        self.tts = tts
        self.config = config
        self.pa_sink = pa_sink
        self.console = console
        self.on_nudge = on_nudge  # callback(msg: str) for web UI broadcast

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

    @property
    def is_excused(self) -> bool:
        return time.monotonic() < self._excused_until

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

        Actions: "start", "stop"
        Excuse is no longer detected here — it's determined from the VLM response.
        """
        lower = text.lower().replace("'", "").replace("\u2019", "")

        for phrase in self.config.start_phrases:
            if phrase in lower:
                return "start"

        for phrase in self.config.stop_phrases:
            if phrase in lower:
                return "stop"

        return None

    def check_response_for_excuse(self, vlm_response: str) -> bool:
        """Check if the VLM's response indicates it granted a pause.

        Called after the VLM responds during active focus mode.
        If the VLM said something like 'go ahead' or 'take your time',
        it means it decided the user's reason is valid — auto-pause.
        """
        if not self._active:
            return False
        if _GRANT_PATTERNS.search(vlm_response):
            self.excuse()
            return True
        return False

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

            # Grab frames from ring buffer and downscale for faster inference
            now = time.monotonic()
            frames = self._grab_small_frames(now)

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

    def _grab_small_frames(self, now: float) -> list[str]:
        """Grab frames from ring buffer, downscale to reduce token usage."""
        with self.camera._lock:
            candidates = [
                (t, f) for t, f in self.camera._ring
                if (now - self.config.check_interval) <= t <= now
            ]

        if not candidates:
            with self.camera._lock:
                if self.camera._ring:
                    candidates = [(self.camera._ring[-1][0], self.camera._ring[-1][1])]
                else:
                    return []

        # Evenly sample across the window
        max_frames = self.config.frames
        if len(candidates) <= max_frames:
            selected = [f for _, f in candidates]
        else:
            step = len(candidates) / max_frames
            selected = [candidates[int(i * step)][1] for i in range(max_frames)]

        # Downscale and re-encode as small JPEG
        result = []
        for frame in selected:
            small = cv2.resize(frame, (_DISTRACTION_WIDTH, _DISTRACTION_HEIGHT))
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, _DISTRACTION_JPEG_QUALITY])
            if ok:
                result.append(base64.b64encode(jpg.tobytes()).decode("ascii"))
        return result

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

            # Extract just the first word — VLM sometimes adds descriptions
            first_word = full_response.strip().split()[0].upper() if full_response.strip() else ""
            result = "DISTRACTED" if "DISTRACT" in first_word else "FOCUSED"
            if self.console:
                self.console.print(f"  [dim]Distraction check: {result}[/dim]")
            return result
        except Exception as e:
            if self.console:
                self.console.print(f"  [dim]Distraction check error: {e}[/dim]")
            return "ERROR"

    def _nudge(self):
        """Speak a random nudge message via TTS and notify web UI."""
        self._last_nudge = time.monotonic()
        msg = random.choice(self.config.nudge_messages)

        if self.console:
            self.console.print(f"  [yellow]Nudge:[/yellow] {msg}")

        if self.on_nudge:
            try:
                self.on_nudge(msg)
            except Exception:
                pass

        if self.tts:
            try:
                result = self.tts.synthesize(msg)
                if result.get("audio") is not None:
                    play_audio(result["audio"], result["sample_rate"], sink=self.pa_sink)
            except Exception:
                pass
