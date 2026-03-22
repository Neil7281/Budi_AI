#!/bin/bash
# =============================================================================
# Safety Pipeline — End-to-End Test on Reachy Mini Jetson
# =============================================================================
#
# Prerequisites:
#   - Main VLM server running on port 8080 (Cosmos-Reason2-2B)
#   - USB camera connected
#   - Reachy Mini powered on
#
# This script walks through 4 stages:
#   1. Launch a dedicated safety VLM on port 8081
#   2. Verify both servers are healthy
#   3. Run a standalone safety-only smoke test (no mic/TTS needed)
#   4. Enable safety in config and launch the full pipeline
#
# Usage:
#   chmod +x test_safety_pipeline.sh
#   ./test_safety_pipeline.sh          # full walkthrough
#   ./test_safety_pipeline.sh --skip-server  # skip step 1 if server already running
# =============================================================================

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SAFETY_PORT=8081
MAIN_PORT=8080
SAFETY_CONTAINER="assistant-safety"

# Recommended small VLMs for safety (pick one via SAFETY_MODEL env var):
#   - ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0   (~266MB, fastest)
#   - ggml-org/SmolVLM2-500M-Video-Instruct-GGUF:Q8_0   (~520MB, good balance)
#   - ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M       (~1.6GB, best quality)
SAFETY_MODEL="${SAFETY_MODEL:-ggml-org/SmolVLM2-500M-Video-Instruct-GGUF:Q8_0}"

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Safety Pipeline — End-to-End Test"
echo "============================================"
echo ""

# ── Step 1: Launch safety VLM server ────────────────────────────
if [ "$1" != "--skip-server" ]; then
    info "Step 1: Launching dedicated safety VLM on port ${SAFETY_PORT}..."
    info "Model: ${SAFETY_MODEL}"
    echo ""

    # Use CPU-only (ngl=0) to avoid GPU contention with main VLM
    # If you have enough VRAM, change -ngl to 999 for GPU
    if [ "$(docker ps -aq -f name=^${SAFETY_CONTAINER}$)" ]; then
        info "Stopping existing ${SAFETY_CONTAINER}..."
        docker stop "$SAFETY_CONTAINER" > /dev/null 2>&1 || true
        docker rm "$SAFETY_CONTAINER" > /dev/null 2>&1 || true
    fi

    HF_CACHE="$HOME/.cache/huggingface"
    mkdir -p "$HF_CACHE"

    docker run -d \
        --name "$SAFETY_CONTAINER" \
        --runtime=nvidia \
        -p "${SAFETY_PORT}:8080" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        ghcr.io/nvidia-ai-iot/llama_cpp:b8095-r36.4-tegra-aarch64-cu126-22.04 \
        llama-server \
        -hf "$SAFETY_MODEL" \
        --host 0.0.0.0 --port 8080 \
        -ngl 0 -c 2048 -np 1

    ok "Container '${SAFETY_CONTAINER}' started"
    echo ""
    echo "  API  : http://localhost:${SAFETY_PORT}/v1/chat/completions"
    echo "  Logs : docker logs -f ${SAFETY_CONTAINER}"
    echo "  Stop : docker stop ${SAFETY_CONTAINER}"
    echo ""

    info "Waiting for safety VLM to load (first run downloads the model, may take a few minutes)..."
    for i in $(seq 1 180); do
        if curl -s "http://localhost:${SAFETY_PORT}/v1/models" > /dev/null 2>&1; then
            ok "Safety VLM server ready after ${i}s"
            break
        fi
        # Check if container died
        if ! docker ps -q -f name=^${SAFETY_CONTAINER}$ > /dev/null 2>&1 || [ -z "$(docker ps -q -f name=^${SAFETY_CONTAINER}$)" ]; then
            fail "Container '${SAFETY_CONTAINER}' exited unexpectedly."
            echo "  Check logs: docker logs ${SAFETY_CONTAINER}"
            docker logs --tail 10 "$SAFETY_CONTAINER" 2>&1 | sed 's/^/  /'
            exit 1
        fi
        if [ "$i" = "180" ]; then
            fail "Safety VLM did not start in 180s. Check: docker logs -f ${SAFETY_CONTAINER}"
            exit 1
        fi
        # Print progress every 15s
        if [ $((i % 15)) = "0" ]; then
            info "  Still waiting... (${i}s elapsed)"
        fi
        sleep 1
    done
else
    info "Step 1: Skipped (--skip-server)"
fi

echo ""

# ── Step 2: Health check both servers ───────────────────────────
info "Step 2: Checking server health..."

# Main VLM
if curl -s "http://localhost:${MAIN_PORT}/v1/models" > /dev/null 2>&1; then
    MAIN_MODEL=$(curl -s "http://localhost:${MAIN_PORT}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
    ok "Main VLM (port ${MAIN_PORT}): ${MAIN_MODEL}"
else
    fail "Main VLM not running on port ${MAIN_PORT}!"
    echo "  Start it with:  NP=1 ./run_llama_cpp.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M"
    exit 1
fi

# Safety VLM
if curl -s "http://localhost:${SAFETY_PORT}/v1/models" > /dev/null 2>&1; then
    SAFETY_MODEL_ID=$(curl -s "http://localhost:${SAFETY_PORT}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
    ok "Safety VLM (port ${SAFETY_PORT}): ${SAFETY_MODEL_ID}"
else
    fail "Safety VLM not running on port ${SAFETY_PORT}!"
    echo "  Start it with:  PORT=8081 NAME=assistant-safety ./run_llama_cpp.sh ${SAFETY_MODEL}"
    exit 1
fi

echo ""

# ── Step 3: Standalone safety inference test ────────────────────
info "Step 3: Testing safety VLM inference (no camera needed)..."

# Send a test request with a tiny 1x1 white JPEG (base64)
# This just verifies the VLM can handle image+text and respond
TEST_IMG="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAFBABAAAAAAAAAAAAAAAAAAAACf/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="

RESPONSE=$(curl -s "http://localhost:${SAFETY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Is there any safety threat in this image? Reply SAFE or THREAT:severity:description"},
                {"type": "image_url", "image_url": {"url": "'"${TEST_IMG}"'"}}
            ]}
        ],
        "max_tokens": 32,
        "temperature": 0.1
    }' 2>&1)

if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['choices'][0]['message']['content']; print(f'  VLM response: {c}')" 2>/dev/null; then
    ok "Safety VLM inference works"
else
    warn "Safety VLM response unexpected (may not support vision):"
    echo "  $RESPONSE" | head -5
    echo ""
    warn "If your model doesn't support images, try a VLM like moondream2 or SmolVLM."
fi

echo ""

# ── Step 4: Python-level SafetyMonitor unit test ────────────────
info "Step 4: Running SafetyMonitor Python smoke test..."

venv/bin/python3 << 'PYEOF'
import time, re, sys

class MockCamera:
    def get_speech_frames(self, speech_start, speech_end, max_frames):
        return ["fake_b64"]

class MockTTS:
    synthesized = []
    def synthesize(self, text):
        self.synthesized.append(text)
        return {"audio": None, "sample_rate": 24000}

class MockConsole:
    def print(self, *a, **kw):
        msg = re.sub(r'\[.*?\]', '', ' '.join(str(x) for x in a)).strip()
        if msg:
            print(f"    {msg}")

from app.config import SafetyConfig
from app.safety import SafetyMonitor

tts = MockTTS()
config = SafetyConfig(enabled=True, check_interval=1.0, cooldown=2.0)
mon = SafetyMonitor(camera=MockCamera(), tts=tts, config=config, console=MockConsole())

# Test response parsing
def fake(text):
    def gen(**kw):
        yield (text, {})
    return gen

mon._llm._loaded = True

tests = [
    ("SAFE",                                 "SAFE",     ""),
    ("THREAT:CRITICAL:fire in the room",     "CRITICAL", "fire in the room"),
    ("THREAT:WARNING:wet floor ahead",       "WARNING",  "wet floor ahead"),
    ("I see a FIRE burning",                 "WARNING",  ""),     # keyword fallback
    ("No issues here",                       "SAFE",     ""),
]

passed = 0
for resp, exp_sev, exp_desc in tests:
    mon._llm.generate_stream = fake(resp)
    sev, desc = mon._query_safety_vlm(["frame"])
    if sev == exp_sev:
        print(f"  ✓ '{resp[:40]}' → {sev}")
        passed += 1
    else:
        print(f"  ✗ '{resp[:40]}' → expected {exp_sev}, got {sev}")

# Test alert triggers TTS
alerts = []
mon.on_alert = lambda m: alerts.append(m)
mon._trigger_alert("CRITICAL", "fire detected")
if tts.synthesized and "fire detected" in tts.synthesized[-1]:
    print(f"  ✓ TTS alert spoken: {tts.synthesized[-1][:60]}...")
    passed += 1
else:
    print(f"  ✗ TTS alert not spoken")

if alerts and alerts[-1]["severity"] == "critical":
    print(f"  ✓ WebSocket callback fired (type={alerts[-1]['type']})")
    passed += 1
else:
    print(f"  ✗ WebSocket callback missing")

# Thread lifecycle
mon.start()
assert mon._running and mon._thread.is_alive()
print(f"  ✓ Background thread started")
passed += 1

mon.stop()
assert not mon._running
print(f"  ✓ Background thread stopped")
passed += 1

total = len(tests) + 4
print(f"\n  Result: {passed}/{total} tests passed")
if passed == total:
    print("  ✅ ALL PASSED")
    sys.exit(0)
else:
    print("  ❌ SOME FAILED")
    sys.exit(1)
PYEOF

PY_EXIT=$?
if [ "$PY_EXIT" = "0" ]; then
    ok "Python smoke test passed"
else
    fail "Python smoke test failed"
    exit 1
fi

echo ""

# ── Step 5: Live integration test ───────────────────────────────
info "Step 5: Testing live safety VLM with real camera frame..."

venv/bin/python3 << 'PYEOF'
import sys
sys.path.insert(0, ".")

from app.config import Config
from app.llm import LLM

config = Config.load()

# Connect to safety VLM
safety_llm = LLM(
    model="",
    base_url=config.safety.model_endpoint,
    backend=config.safety.model_backend,
    max_tokens=64,
    temperature=0.1,
    timeout=30.0,
)

if not safety_llm.load():
    print("  ✗ Cannot connect to safety VLM")
    sys.exit(1)

print(f"  Connected to safety VLM: {safety_llm.model}")

# Try to grab a real camera frame
try:
    from app.camera import Camera
    cam = Camera(
        device=config.vision.camera_device,
        width=config.vision.width,
        height=config.vision.height,
        jpeg_quality=config.vision.jpeg_quality,
        capture_fps=config.vision.capture_fps,
    )
    if cam.start():
        import time
        time.sleep(1)  # let ring buffer fill
        now = time.monotonic()
        frames = cam.get_speech_frames(now - 1, now, max_frames=1)
        cam.close()

        if frames:
            print(f"  Captured {len(frames)} frame(s), sending to safety VLM...")
            full = ""
            for chunk in safety_llm.generate_stream(
                prompt=config.safety.prompt,
                images_b64=frames,
                max_tokens=64,
                temperature=0.1,
            ):
                content, meta = chunk if isinstance(chunk, tuple) else (chunk, {})
                if content:
                    full += content
            print(f"  Safety VLM says: {full.strip()}")
            print("  ✅ Live camera → safety VLM pipeline works!")
        else:
            print("  ⚠ No frames captured from ring buffer")
    else:
        print("  ⚠ Camera not available, skipping live test")
except Exception as e:
    print(f"  ⚠ Camera test skipped: {e}")

sys.exit(0)
PYEOF

echo ""

# ── Step 6: Instructions for full pipeline test ─────────────────
echo "============================================"
echo "  Ready for Full Pipeline Test"
echo "============================================"
echo ""
info "To test with the full conversation pipeline:"
echo ""
echo "  1. Enable safety in config/settings.yaml:"
echo "     ${CYAN}safety:"
echo "       enabled: true${NC}"
echo ""
echo "  2. Run the full pipeline:"
echo "     ${CYAN}venv/bin/python3 run_web_vision_chat.py${NC}"
echo "     or"
echo "     ${CYAN}venv/bin/python3 run_vision_chat.py${NC}"
echo ""
echo "  3. Look for this in the startup output:"
echo "     ${GREEN}✓ Safety VLM (model_name @ http://localhost:8081)${NC}"
echo "     ${GREEN}✓ Safety monitor active${NC}"
echo ""
echo "  4. Test safety alerts by:"
echo "     - Holding a lighter or match in view of the camera"
echo "     - Showing a picture of fire on your phone"
echo "     - Pretending to fall/collapse"
echo "     - The robot should INTERRUPT any current speech and say the alert"
echo ""
echo "  5. Check safety monitor logs in terminal:"
echo "     ${CYAN}Safety check: SAFE${NC}             (normal, every 3s)"
echo "     ${YELLOW}⚠ SAFETY WARNING: ...${NC}        (threat detected)"
echo "     ${RED}⚠ SAFETY CRITICAL: ...${NC}       (urgent threat)"
echo ""
echo "  6. Stop the safety VLM when done:"
echo "     ${CYAN}docker stop ${SAFETY_CONTAINER}${NC}"
echo ""
