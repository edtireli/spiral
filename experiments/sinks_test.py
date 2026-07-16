"""Tier-3 experiment: what survives context overflow? (attention-sinks probe)

Plants three needles (early / middle / late) in a prompt ~3x the context
window, then asks for each — once with default overflow handling, once with
num_keep (ollama's pin-the-first-N-tokens knob: the poor man's attention sink).

Prediction: LATE always survives (the window keeps the tail), EARLY survives
only when pinned, MIDDLE always dies — quantifying exactly what "infinite
streaming" forgets. Uses the 1.3GB llama3.2:1b so it can run anytime.

    python experiments/sinks_test.py
"""
from __future__ import annotations

import httpx

MODEL = "llama3.2:1b"
NUM_CTX = 1024
URL = "http://localhost:11434/api/chat"

NEEDLES = {
    "early": "The vault code is 7431.",
    "middle": "The password for the archive is FERNWEH.",
    "late": "The courier arrives at 6pm sharp.",
}
QUESTIONS = {
    "early": "What is the vault code?",
    "middle": "What is the archive password?",
    "late": "At what time does the courier arrive?",
}
ANSWERS = {"early": "7431", "middle": "fernweh", "late": "6"}

FILLER = ("The municipal record continues with routine entries about weather, "
          "deliveries, and maintenance schedules. ")


def ask(client: httpx.Client, question: str, num_keep: int) -> str:
    pad = FILLER * 60  # ~1400 tok per pad → prompt ≈ 3x the window
    prompt = (f"{NEEDLES['early']}\n{pad}\n{NEEDLES['middle']}\n{pad}\n"
              f"{NEEDLES['late']}\nAnswer briefly: {question}")
    r = client.post(URL, json={
        "model": MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": NUM_CTX, "num_predict": 30,
                    "temperature": 0.0, "num_keep": num_keep},
    }, timeout=120)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def trial(client: httpx.Client, label: str, num_keep: int) -> None:
    print(f"\n── {label} ──")
    for pos, q in QUESTIONS.items():
        text = ask(client, q, num_keep)
        got = ANSWERS[pos] in text.lower()
        print(f"  {pos:7s} needle: {'✓ SURVIVED ' if got else '✗ forgotten'}  → {' '.join(text.split())[:60]}")


def main() -> int:
    with httpx.Client() as client:
        try:
            client.get("http://localhost:11434/api/version", timeout=5)
        except Exception:
            print("ollama unreachable")
            return 1
        print(f"window {NUM_CTX} tok · prompt ≈ 3x that · {MODEL}")
        trial(client, "default overflow (no sink)", num_keep=0)
        trial(client, "num_keep=64 — first tokens pinned (attention-sink analog)", num_keep=64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
