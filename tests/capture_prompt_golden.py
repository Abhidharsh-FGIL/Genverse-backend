"""HISTORICAL: how tests/golden/archive/prompts_v0.json was recorded.

It was run once against the ORIGINAL code, before the Phase 1 prompt refactor,
and backed the Phase 1 golden test (removed in Phase 2, when the wording changed
on purpose). The archive is kept for reference. Running this now would capture
the CURRENT wording, so it writes to a new file instead of overwriting v0:
    .venv/bin/python tests/capture_prompt_golden.py
Not collected by pytest (filename does not start with test_).
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from _prompt_harness import build_combos, capture  # noqa: E402


async def main() -> None:
    out = []
    mp = pytest.MonkeyPatch()
    try:
        for combo in build_combos():
            rec = await capture(mp, combo["context"], combo["settings"])
            out.append({**combo, "captured": rec})
    finally:
        mp.undo()
    path = Path(__file__).parent / "golden" / "prompts_current.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {len(out)} combos to {path}")


if __name__ == "__main__":
    asyncio.run(main())
