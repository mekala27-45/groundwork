"""Records docs/demo.gif by driving a real browser against a real running
instance of this app, the same three-workspace, no-setup demo the README
points at, not a mockup and not a manually captured screen recording.

Prerequisites, all real, none stubbed: the API running at
GROUNDWORK_API_URL (`make dev` migrations plus
`uv run uvicorn groundwork_api.app:app --app-dir packages/api/src` from
the repo root, or `docker compose up`), the demo workspaces seeded
(`uv run python scripts/seed_demo_workspaces.py`), and the web app
running at GROUNDWORK_WEB_URL (`cd web && npm run dev`). Both default to
the same local addresses CONTRIBUTING.md's own setup instructions bring
up. The web app's origin must be one `Settings.cors_origins` actually
allows (`http://localhost:3000` by default, not `http://127.0.0.1:3000`,
which is a different origin to a browser's CORS check even though both
reach the same server); using anything else here silently reproduces the
exact bug this script's own first real run found, below.

Deliberately not a `uv run` script: it needs playwright and a real
Chromium binary, a heavy, browser-download-on-install dependency this
workspace's own pyproject.toml does not carry, since nothing at runtime
or in the test suite needs it, so this runs against a plain system
Python instead. Run with:

    pip install playwright && playwright install chromium
    python3 scripts/record_demo.py

and ffmpeg on the PATH. Set PLAYWRIGHT_CHROMIUM_PATH if the Chromium
`playwright install` fetches is not at the default path this script
assumes (/opt/pw-browsers/chromium, this build's own sandbox's
pre-fetched copy; a fresh `playwright install chromium` reports its own
install path directly).

A real bug this script found, not a hypothetical one used to justify
writing it after the fact: the first real recording landed on a /trace
page reading "Could not load this trace, turn not found" for a turn the
chat page had just linked to, immediately after the API itself returned
201 Created for it. groundwork_api.routers.conversations.ask_question
called chat.ask(), which only flushes its new Turn by design (so
scripts/run_eval.py can batch many calls under one commit), and never
committed itself, unlike create_conversation right above it in the same
file. Every existing test drove its HTTP calls through one shared test
session (conftest.py's client fixture), which made a flush indistinguishable
from a commit in every test that already existed. Nothing hand-written
would have caught this the way a real browser, making real separate
requests against a real running server, did on the very first attempt to
record a demo of the finished app. Fixed by committing in ask_question,
covered by a regression test that opens a second, genuinely independent
database connection to prove durability
(test_ask_question_actually_commits_so_a_separate_connection_can_see_it).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

API_URL = os.environ.get("GROUNDWORK_API_URL", "http://localhost:8000")
WEB_URL = os.environ.get("GROUNDWORK_WEB_URL", "http://localhost:3000")
CHROMIUM_PATH = os.environ.get("PLAYWRIGHT_CHROMIUM_PATH", "/opt/pw-browsers/chromium")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "docs" / "demo.gif"

DEMO_QUESTION = "How long does a standard Meridian coaching session run?"


async def _record(video_dir: Path) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROMIUM_PATH, headless=True)
        context = await browser.new_context(
            viewport={"width": 1180, "height": 760},
            record_video_dir=str(video_dir),
            record_video_size={"width": 1180, "height": 760},
        )
        page = await context.new_page()

        await page.goto(f"{WEB_URL}/chat/", wait_until="networkidle")
        await page.wait_for_timeout(1200)

        await page.get_by_role("button", name="Meridian Coaching").click()
        await page.wait_for_timeout(1500)

        question_box = page.get_by_placeholder("Ask a question...")
        await question_box.click()
        await question_box.press_sequentially(DEMO_QUESTION, delay=35)
        await page.wait_for_timeout(500)

        await page.get_by_role("button", name="Ask", exact=True).click()
        await page.get_by_text("View full trace").wait_for(timeout=20000)
        await page.wait_for_timeout(2800)

        await page.get_by_text("View full trace").click()
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2500)
        await page.mouse.wheel(0, 500)
        await page.wait_for_timeout(2200)
        await page.mouse.wheel(0, 500)
        await page.wait_for_timeout(2200)

        await page.goto(f"{WEB_URL}/eval/", wait_until="networkidle")
        await page.wait_for_timeout(2500)
        await page.mouse.wheel(0, 500)
        await page.wait_for_timeout(2500)
        await page.mouse.wheel(0, 600)
        await page.wait_for_timeout(2500)

        await context.close()
        await browser.close()


def _convert_to_gif(video_dir: Path, output_path: Path) -> None:
    webm_files = list(video_dir.glob("*.webm"))
    if not webm_files:
        raise RuntimeError(f"no recording found in {video_dir}")
    webm_path = webm_files[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(webm_path),
            "-vf",
            "fps=10,scale=820:-1:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="groundwork-demo-") as tmp:
        video_dir = Path(tmp)
        asyncio.run(_record(video_dir))
        _convert_to_gif(video_dir, OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"wrote {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - a CLI entry point's own top level
        print(f"record_demo failed: {exc}", file=sys.stderr)
        print(
            f"is the API running at {API_URL} and the web app at {WEB_URL}, "
            "with the demo workspaces seeded?",
            file=sys.stderr,
        )
        sys.exit(1)
