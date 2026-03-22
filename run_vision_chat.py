#!/usr/bin/env python3
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

"""
Vision Chat — speak + see, the VLM describes what it sees.
Mic -> Silero/energy VAD -> [camera capture] -> STT -> VLM (text + images) -> TTS -> Speaker

Usage:
  python3 run_vision_chat.py
"""

import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.audio import find_alsa_device
from app.stt import STT
from app.llm import LLM
from app.tts import create_tts
from app.camera import Camera
from app.pipeline import (
    SAMPLE_RATE, MicRecorder, warmup_stt, vad_loop, stream_and_speak, load_silero, play_audio,
)
from app.reachy import kill_stale_camera_holders, connect as connect_reachy
from app.emotion import EmotionDetector
from app.movements import MovementController
from app.distraction import DistractionMonitor
from app.scheduler import Scheduler, parse_reminder, parse_cancel
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    config = Config.load()

    console.print(Panel.fit(
        "[bold cyan]Vision Chat[/bold cyan]\n"
        "Speak anytime — camera captures when you speak\n"
        "[dim]Ctrl-C to quit[/dim]",
        border_style="cyan",
    ))

    # ── Reachy Mini ──────────────────────────────────────────────
    reachy = connect_reachy(config, console)

    # ── Audio setup ──────────────────────────────────────────────
    result = find_alsa_device(name_hint=config.audio.input_device or "Reachy Mini Audio")
    if not result:
        console.print("[red]No mic found![/red]")
        return
    card, dev, mic_name = result
    hw = f"hw:{card},{dev}"
    console.print(f"  Mic: {hw} ({mic_name})")

    # ── Camera setup (background ring buffer) ────────────────────
    kill_stale_camera_holders(config.vision.camera_device, console)

    cam = Camera(
        device=config.vision.camera_device,
        width=config.vision.width,
        height=config.vision.height,
        jpeg_quality=config.vision.jpeg_quality,
        capture_fps=config.vision.capture_fps,
    )
    if cam.start():
        console.print(
            f"  ✓ Camera /dev/video{config.vision.camera_device} "
            f"({config.vision.width}x{config.vision.height}, "
            f"{config.vision.capture_fps} fps ring buffer)"
        )
    else:
        console.print("[red]  ✗ Camera not found! Check USB webcam.[/red]")
        return

    # ── Pre-declare variables for cleanup closure ───────────────
    mic = None
    stt = None
    llm = None
    tts = None
    silero_model = None

    # ── Cleanup handler ──────────────────────────────────────────
    _cleanup_done = threading.Event()

    def _do_cleanup():
        if _cleanup_done.is_set():
            return
        _cleanup_done.set()
        console.print("\n[yellow]Shutting down...[/yellow]")
        if mic:
            try:
                mic.stop()
            except Exception:
                pass
        cam.close()
        if reachy and config.reachy.sleep_on_exit:
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except OSError:
                pass
            try:
                console.print("  Putting Reachy Mini to sleep...")
                reachy.goto_sleep()
                time.sleep(0.5)
                reachy.disable_motors()
                time.sleep(0.3)
            except Exception as e:
                console.print(f"  [dim]Sleep failed: {e}[/dim]")

    def _sig_cleanup(signum=None, frame=None):
        _do_cleanup()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sig_cleanup)
    signal.signal(signal.SIGTSTP, _sig_cleanup)
    signal.signal(signal.SIGTERM, _sig_cleanup)
    signal.signal(signal.SIGHUP, _sig_cleanup)

    # ── Load models ──────────────────────────────────────────────
    console.print("\n[bold]Loading...[/bold]")

    stt = STT(
        model=config.stt.model, device=config.stt.device,
        compute_type=config.stt.compute_type, language=config.stt.language,
        beam_size=config.stt.beam_size,
    )
    stt.load()
    console.print(f"  ✓ STT (faster-whisper, {config.stt.model})")
    console.print("    CUDA warmup...", end=" ")
    console.print(f"done ({warmup_stt(stt):.1f}s)")

    if config.vad.use_silero:
        silero_model = load_silero(console)
    else:
        console.print("  [dim]Silero VAD disabled, using energy-only VAD[/dim]")

    vision_system_prompt = config.vision.system_prompt
    vision_few_shot = config.vision.few_shot or []
    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=vision_system_prompt,
    )
    llm.load()
    console.print(f"  ✓ VLM ({llm.model})")

    tts = create_tts(
        voice=config.tts.voice, speed=config.tts.speed, lang=config.tts.lang,
        backend=config.tts.backend,
    )
    tts = tts if tts.load() else None
    if tts:
        console.print(f"  ✓ TTS ({tts.backend_name}, {tts.voice})")
    else:
        console.print("  ⚠ TTS unavailable")

    emotion_detector = None
    mover = None
    if config.emotion.enabled:
        emotion_detector = EmotionDetector()
        if emotion_detector.load():
            console.print("  ✓ Emotion (distilbert-sst2, CPU)")
            if reachy:
                mover = MovementController(reachy, config.reachy.antenna_rest_position)
                console.print("  ✓ Emotion movements enabled")
        else:
            console.print("  ⚠ Emotion detector unavailable")
            emotion_detector = None

    # ── Distraction monitor (activated by voice command) ─────────
    distraction_monitor = None
    if config.distraction.enabled:
        distraction_monitor = DistractionMonitor(
            llm=llm, camera=cam, tts=tts,
            config=config.distraction,
            pa_sink=None,  # pa_sink set after mic starts
            console=console,
        )
        console.print("  ✓ Distraction monitor ready (say 'focus mode' to activate)")

    # ── Scheduler (voice-activated reminders) ────────────────────
    scheduler = Scheduler(tts=tts, pa_sink=None, console=console)
    console.print("  ✓ Scheduler ready (say 'remind me in X minutes to...')")

    # ── Start mic ────────────────────────────────────────────────
    effective_chunk_ms = 32 if silero_model else config.vad.chunk_ms
    mic = MicRecorder(console, chunk_ms=effective_chunk_ms)
    if not mic.start(hw, config.audio.input_device or "Reachy Mini Audio"):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        cam.close()
        return

    if distraction_monitor:
        distraction_monitor.pa_sink = mic.pa_sink
    scheduler.pa_sink = mic.pa_sink

    n_frames = config.vision.frames
    n_fewshot = len(vision_few_shot) // 2
    console.print(
        f"\n[green bold]Ready — speak anytime! "
        f"({config.vision.capture_fps} fps, {n_frames} frame{'s' if n_frames > 1 else ''} "
        f"per query{f', {n_fewshot} few-shot pairs' if n_fewshot else ''})[/green bold]\n"
    )

    # ── Main loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            # Pause distraction checks while handling conversation
            if distraction_monitor:
                distraction_monitor.pause()

            t_cam = time.perf_counter()
            captured_frames = cam.get_speech_frames(
                speech_start=segment.start_time,
                speech_end=segment.end_time,
                max_frames=n_frames,
            )
            dt_cam = time.perf_counter() - t_cam

            t_stt = time.perf_counter()
            result = stt.transcribe(segment.audio, sample_rate=SAMPLE_RATE)
            text = result.get("text", "").strip()
            dt_stt = time.perf_counter() - t_stt

            if not text:
                err = result.get("error", "")
                console.print(
                    f"[dim]  (not recognized — {segment.duration:.1f}s, "
                    f"rms={segment.rms:.4f}{', err='+err if err else ''})[/dim]"
                )
                if distraction_monitor:
                    distraction_monitor.resume()
                mic.resume()
                continue

            # ── Check for cancel reminders ───────────────────────────
            if parse_cancel(text):
                count = scheduler.cancel_recurring()
                if count > 0:
                    confirm = f"Done, I cancelled {count} recurring reminder{'s' if count != 1 else ''}."
                else:
                    confirm = "You don't have any active recurring reminders to cancel."
                console.print(f'  [green]You:[/green] "{text}"')
                console.print(f"  [magenta]Assistant:[/magenta] {confirm}")
                if tts:
                    r = tts.synthesize(confirm)
                    if r.get("audio") is not None:
                        play_audio(r["audio"], r["sample_rate"], sink=mic.pa_sink)
                if distraction_monitor:
                    distraction_monitor.resume()
                mic.resume()
                continue

            # ── Check for reminders (parsed + sent to VLM for context) ──
            reminder_parsed = parse_reminder(text)
            if reminder_parsed:
                delay, task, recurring = reminder_parsed
                scheduler.add(delay, task, recurring=recurring)

            # ── Check for focus mode triggers (before filler filter) ──
            distraction_action = None
            if distraction_monitor:
                distraction_action = distraction_monitor.check_text(text)
                if distraction_action == "start":
                    distraction_monitor.start()
                elif distraction_action == "stop":
                    distraction_monitor.stop()

            word_count = len(text.split())
            if word_count <= 2 and "?" not in text and not distraction_action and not reminder_parsed:
                console.print(f"[dim]  (skipped filler: \"{text}\")[/dim]")
                if distraction_monitor:
                    distraction_monitor.resume()
                mic.resume()
                continue

            emotion_tag = ""
            if emotion_detector:
                emo = emotion_detector.detect(text)
                moved = mover.react(emo.emotion, emo.confidence) if mover else False
                emotion_tag = (
                    f" | {emo.emotion.value} ({emo.confidence:.0%}, {emo.inference_ms:.0f}ms)"
                    f"{'*' if moved else ''}"
                )

            n_imgs = len(captured_frames)
            console.print(
                f'  [green]You:[/green] "{text}" '
                f'[dim]({n_imgs} frame{"s" if n_imgs != 1 else ""} captured)[/dim]'
            )

            console.print("  [magenta]Assistant:[/magenta] ", end="")
            sys.stdout.flush()

            # Add reminder context so VLM knows a reminder was just set
            llm_prompt = text
            if reminder_parsed:
                delay, task, recurring = reminder_parsed
                from app.scheduler import _format_time
                time_str = _format_time(delay)
                if recurring:
                    llm_prompt = (
                        f"[SYSTEM: The user just asked you to set a recurring reminder. "
                        f"You have already set a recurring reminder every {time_str} to '{task}'. "
                        f"Briefly confirm this naturally. Keep it short and friendly.] {text}"
                    )
                else:
                    llm_prompt = (
                        f"[SYSTEM: The user just asked you to set a reminder. "
                        f"You have already set a reminder for {time_str} from now to '{task}'. "
                        f"Briefly confirm this naturally. Keep it short and friendly.] {text}"
                    )

            if distraction_monitor and distraction_monitor.is_active:
                llm_prompt = f"[Focus mode is ON. The user is being monitored for distractions.] {text}"

            full_resp, dt_llm, ttft = stream_and_speak(
                llm, tts, llm_prompt, vision_system_prompt, mic.pa_sink,
                images_b64=captured_frames if captured_frames else None,
                few_shot=vision_few_shot if vision_few_shot else None,
                first_chunk_words=config.tts.first_chunk_words,
                max_chunk_words=config.tts.max_chunk_words,
            )
            console.print()

            timing = f"  [dim]STT {dt_stt:.1f}s | CAM {dt_cam*1000:.0f}ms ({n_imgs} img from buf)"
            if ttft is not None:
                toks = len(full_resp.split())
                timing += f" | TTFT {ttft:.1f}s | VLM {dt_llm:.1f}s ~{toks/(dt_llm or 1):.0f}w/s"
            else:
                timing += " | VLM no response"
            timing += emotion_tag
            if distraction_monitor and distraction_monitor.is_active:
                timing += " | [cyan]FOCUS[/cyan]"
            timing += "[/dim]"
            console.print(timing)

            # Resume distraction checks after conversation turn
            if distraction_monitor:
                distraction_monitor.resume()
            mic.resume()

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        pass

    _do_cleanup()
    if distraction_monitor:
        distraction_monitor.stop()
    if mover:
        mover.reset()
    try:
        if stt:
            stt.unload()
        if llm:
            llm.unload()
        if tts:
            tts.unload()
        if emotion_detector:
            emotion_detector.unload()
    except Exception:
        pass
    console.print("[yellow]Goodbye![/yellow]")
    os._exit(0)


if __name__ == "__main__":
    main()
