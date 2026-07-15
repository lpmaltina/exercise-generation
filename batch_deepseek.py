#!/usr/bin/env python3
"""
Batch-process texts with DeepSeek V4 Pro (Expert Mode, thinking enabled).

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python batch_deepseek.py input.txt output.txt
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Write your fixed prompt here. It will be prepended to each text.
FIXED_PROMPT = """[PLACEHOLDER PROMPT — replace me]

Below is a text. Do <whatever you want> with it:

"""

# DeepSeek V4 Pro == "Expert Mode" on chat.deepseek.com
MODEL_NAME = "deepseek-v4-pro"
BASE_URL = "https://api.deepseek.com"

# Thinking mode ON, high reasoning effort.
# Change "high" to "max" if you want maximum reasoning (more expensive).
EXTRA_BODY = {
    "thinking": {"type": "enabled"},
    "reasoning_effort": "high",
}

# Separator between texts in the input file: three blank lines
# i.e. four consecutive newlines (\n\n\n\n). We're lenient and also
# accept more than three blank lines.
INPUT_SEPARATOR_RE = re.compile(r"\n\s*\n\s*\n\s*\n+")

# Separator to use between blocks in the output file
OUTPUT_SEPARATOR = "\n\n\n\n"

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_texts(path: Path) -> list[str]:
    """Read the input file and split it into individual texts."""
    raw = path.read_text(encoding="utf-8")
    chunks = INPUT_SEPARATOR_RE.split(raw)
    # Strip and drop empties
    return [c.strip() for c in chunks if c.strip()]


def wrap_original(text: str) -> str:
    """Wrap the original text in the <details> block."""
    return f"<details>\n  <summary>original</summary>{text}</details>"


def call_deepseek(client: OpenAI, user_content: str) -> str:
    """Send one request to DeepSeek and return the assistant reply."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": user_content}],
                extra_body=EXTRA_BODY,
            )
            return response.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(
                f"  ! API error on attempt {attempt}/{MAX_RETRIES}: {e}",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Giving up after {MAX_RETRIES} attempts: {last_error}")


def build_output_block(model_response: str, wrapped_original: str) -> str:
    """Assemble one output block per the target format."""
    return f"DeepSeek Auto\n\n" f"{model_response}\n\n\n" f" {wrapped_original}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input .txt file")
    parser.add_argument("output", type=Path, help="Output .txt file")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "ERROR: DEEPSEEK_API_KEY not set (env var or --api-key).", file=sys.stderr
        )
        return 2

    if not args.input.is_file():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    texts = load_texts(args.input)
    if not texts:
        print("ERROR: no non-empty texts found in input file.", file=sys.stderr)
        return 1

    print(f"Loaded {len(texts)} text(s) from {args.input}")

    client = OpenAI(api_key=args.api_key, base_url=BASE_URL)

    blocks: list[str] = []
    for i, original in enumerate(texts, start=1):
        print(f"[{i}/{len(texts)}] Sending to {MODEL_NAME} (thinking=on)...")
        wrapped = wrap_original(original)
        user_content = FIXED_PROMPT + original
        response_text = call_deepseek(client, user_content)
        block = build_output_block(response_text, wrapped)
        blocks.append(block)

        # Incremental save so we don't lose progress on a mid-run failure
        args.output.write_text(OUTPUT_SEPARATOR.join(blocks) + "\n", encoding="utf-8")
        print(f"  ok ({len(response_text)} chars)")

    print(f"Done. Wrote {len(blocks)} block(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
