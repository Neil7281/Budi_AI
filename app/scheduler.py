"""Scheduler — voice-activated reminders with VLM context awareness.

User says "remind me in 2 minutes to drink water" → regex parses time → VLM confirms naturally
         "remind me every 30 seconds to drink water" → recurring reminder
When timer fires → TTS speaks the contextual reminder.
"""

import re
import threading
import time
from typing import Optional

from app.pipeline import play_audio


class Reminder:
    def __init__(self, delay: float, message: str, created: float, recurring: bool = False):
        self.delay = delay
        self.message = message
        self.created = created
        self.fire_at = created + delay
        self.fired = False
        self.recurring = recurring

    def reschedule(self):
        """Reschedule a recurring reminder from now."""
        self.fire_at = time.monotonic() + self.delay
        self.fired = False


class Scheduler:
    """Voice-activated reminder scheduler with recurring support."""

    def __init__(self, tts=None, pa_sink=None, console=None, on_reminder=None):
        self.tts = tts
        self.pa_sink = pa_sink
        self.console = console
        self.on_reminder = on_reminder  # callback(msg) for web UI
        self._reminders: list[Reminder] = []
        self._lock = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()

    def add(self, delay_seconds: float, message: str, recurring: bool = False):
        """Add a reminder that fires after delay_seconds. If recurring, it repeats."""
        r = Reminder(delay_seconds, message, time.monotonic(), recurring=recurring)
        with self._lock:
            self._reminders.append(r)
        time_str = _format_time(delay_seconds)
        prefix = "Recurring reminder" if recurring else "Reminder"
        if self.console:
            self.console.print(f"  [cyan]{prefix} set: \"{message}\" {'every' if recurring else 'in'} {time_str}[/cyan]")

    def cancel_recurring(self, keyword: str = None) -> int:
        """Cancel recurring reminders. If keyword given, only those matching."""
        count = 0
        with self._lock:
            kept = []
            for r in self._reminders:
                if r.recurring and (keyword is None or keyword.lower() in r.message.lower()):
                    count += 1
                else:
                    kept.append(r)
            self._reminders = kept
        return count

    def _check_loop(self):
        """Background loop checking for due reminders."""
        while self._alive:
            time.sleep(1)
            now = time.monotonic()
            due = []
            with self._lock:
                for r in self._reminders:
                    if not r.fired and now >= r.fire_at:
                        r.fired = True
                        due.append(r)
                # Reschedule recurring, remove one-shot fired
                for r in due:
                    if r.recurring:
                        r.reschedule()
                self._reminders = [r for r in self._reminders if not r.fired or r.recurring]

            for r in due:
                self._fire(r)

    def _fire(self, reminder: Reminder):
        """Speak the reminder via TTS and notify web UI."""
        msg = f"Hey! Reminder: {reminder.message}"

        if self.console:
            tag = " (recurring)" if reminder.recurring else ""
            self.console.print(f"  [yellow]Reminder{tag}:[/yellow] {msg}")

        if self.on_reminder:
            try:
                self.on_reminder(msg, reminder.recurring)
            except Exception:
                pass

        if self.tts:
            try:
                result = self.tts.synthesize(msg)
                if result.get("audio") is not None:
                    play_audio(result["audio"], result["sample_rate"], sink=self.pa_sink)
            except Exception:
                pass

    def stop(self):
        self._alive = False

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._reminders)


def _format_time(seconds: float) -> str:
    if seconds >= 3600:
        hrs = seconds / 3600
        return f"{hrs:.0f} hour{'s' if hrs != 1 else ''}"
    elif seconds >= 60:
        mins = seconds / 60
        return f"{mins:.0f} minute{'s' if mins != 1 else ''}"
    else:
        return f"{seconds:.0f} second{'s' if seconds != 1 else ''}"


# ── Time parsing (regex-based detection) ──────────────────────────────

_TIME_PATTERNS = [
    # "in 5 minutes", "in 2 mins", "in 30 seconds", "in 1 hour"
    re.compile(r"in\s+(\d+)\s*(second|sec|minute|min|hour|hr)s?", re.I),
    # "after 5 minutes"
    re.compile(r"after\s+(\d+)\s*(second|sec|minute|min|hour|hr)s?", re.I),
    # "2 minutes from now"
    re.compile(r"(\d+)\s*(second|sec|minute|min|hour|hr)s?\s+from\s+now", re.I),
    # "in half an hour", "in half a minute"
    re.compile(r"in\s+half\s+an?\s+(hour|minute)", re.I),
    # "in a minute", "in an hour"
    re.compile(r"in\s+an?\s+(second|minute|hour)", re.I),
]

# Recurring patterns: "every 10 seconds", "every 5 minutes"
_RECURRING_PATTERNS = [
    re.compile(r"every\s+(\d+)\s*(second|sec|minute|min|hour|hr)s?", re.I),
    re.compile(r"every\s+half\s+an?\s+(hour|minute)", re.I),
    re.compile(r"every\s+an?\s+(second|minute|hour)", re.I),
]

_TASK_PATTERNS = [
    # "remind me in 5 minutes to drink water"
    re.compile(r"remind\s+me\s+.*?\s+to\s+(.+)", re.I),
    # "remind me to drink water in 5 minutes"
    re.compile(r"remind\s+me\s+to\s+(.+?)(?:\s+in\s+\d+|\s+after\s+\d+|\s+every\s+\d+|$)", re.I),
    # "in 5 minutes tell me to drink water"
    re.compile(r"(?:in|after|every)\s+\d+\s*\w+\s+(?:tell|remind)\s+me\s+to\s+(.+)", re.I),
    # "set a reminder to drink water"
    re.compile(r"(?:set|create)\s+a?\s*reminder\s+(?:to\s+)?(.+?)(?:\s+in\s+\d+|\s+after\s+\d+|\s+every\s+\d+|$)", re.I),
]

_UNIT_MULTIPLIERS = {
    "second": 1, "sec": 1,
    "minute": 60, "min": 60,
    "hour": 3600, "hr": 3600,
}

# Cancel patterns
_CANCEL_PATTERNS = [
    re.compile(r"(?:stop|cancel|remove|clear|delete)\s+(?:all\s+)?(?:the\s+)?(?:recurring\s+)?reminder", re.I),
    re.compile(r"(?:stop|cancel)\s+reminding\s+me", re.I),
]


def parse_reminder(text: str) -> Optional[tuple[float, str, bool]]:
    """Parse a reminder from user text. Returns (delay_seconds, task, recurring) or None."""
    lower = text.lower()

    # Must contain reminder-related keywords
    if not any(w in lower for w in ["remind", "timer", "alert me", "tell me in", "notify"]):
        return None

    # Check for recurring pattern first
    recurring = False
    delay = None

    for pattern in _RECURRING_PATTERNS:
        m = pattern.search(lower)
        if m:
            recurring = True
            groups = m.groups()
            if len(groups) == 2:
                try:
                    amount = int(groups[0])
                    unit = groups[1].lower().rstrip("s")
                    delay = amount * _UNIT_MULTIPLIERS.get(unit, 60)
                except ValueError:
                    pass
            elif len(groups) == 1:
                unit = groups[0].lower()
                if "half" in m.group():
                    delay = _UNIT_MULTIPLIERS.get(unit, 60) / 2
                else:
                    delay = _UNIT_MULTIPLIERS.get(unit, 60)
            break

    # If not recurring, try one-shot patterns
    if delay is None:
        for pattern in _TIME_PATTERNS:
            m = pattern.search(lower)
            if m:
                groups = m.groups()
                if len(groups) == 2:
                    try:
                        amount = int(groups[0])
                        unit = groups[1].lower().rstrip("s")
                        delay = amount * _UNIT_MULTIPLIERS.get(unit, 60)
                    except ValueError:
                        pass
                elif len(groups) == 1:
                    unit = groups[0].lower()
                    if "half" in m.group():
                        delay = _UNIT_MULTIPLIERS.get(unit, 60) / 2
                    else:
                        delay = _UNIT_MULTIPLIERS.get(unit, 60)
                break

    if delay is None:
        return None

    # Parse task
    task = None
    for pattern in _TASK_PATTERNS:
        m = pattern.search(text)
        if m:
            task = m.group(1).strip().rstrip(".")
            # Clean up time references from the task
            task = re.sub(r"\s*in\s+\d+\s*(second|sec|minute|min|hour|hr)s?\s*", " ", task, flags=re.I).strip()
            task = re.sub(r"\s*after\s+\d+\s*(second|sec|minute|min|hour|hr)s?\s*", " ", task, flags=re.I).strip()
            task = re.sub(r"\s*every\s+\d+\s*(second|sec|minute|min|hour|hr)s?\s*", " ", task, flags=re.I).strip()
            if task:
                break

    if not task:
        task = "check in"

    return (delay, task, recurring)


def parse_cancel(text: str) -> bool:
    """Check if user wants to cancel reminders."""
    for pattern in _CANCEL_PATTERNS:
        if pattern.search(text):
            return True
    return False
