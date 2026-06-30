#!/usr/bin/env python3
"""부트스트랩 ③적재 — 리뷰 통과(STATUS: APPROVED)한 코드를 학습셋으로 적재.

bootstrap/review/{plat}_{key}.py 에서 STATUS APPROVED만 골라:
  - '코드' 마커 아래 코드 추출 → ast.parse 게이트
  - .meta.json의 TC + collect_data.build_sample(학습=추론 포맷) → raw_data/pair_{plat}_{key}.json
  - source=bootstrap, draft_lines/final_lines(트림량) 메타 기록

사용법: python3 bootstrap_ingest.py   (--dry로 미리보기)
이후: python3 prepare_data.py → mlx_lm lora --config config.yaml --train → _validate_6144.py
"""
import re, sys, json, ast, argparse
from pathlib import Path

FT = Path(__file__).parent
sys.path.insert(0, str(FT))
REVIEW = FT / "bootstrap" / "review"
OUT = FT / "raw_data"

from collect_data import build_sample, elements_context, appsettings_context

HEADER = re.compile(r"STATUS:\s*(\w+)\s*(?:\(([^)]*)\))?")  # STATUS: SKIP (사유)
CODE_MARK = "===== 코드"
SKIPLOG = FT / "bootstrap" / "skipped.json"


def extract(py_path: Path):
    text = py_path.read_text(encoding="utf-8")
    first = text.splitlines()[0] if text else ""
    m = HEADER.search(first)
    status = m.group(1).upper() if m else "DRAFT"
    reason = (m.group(2) or "").strip() if m else ""
    # '코드' 마커 줄 이후가 학습 코드
    idx = text.find(CODE_MARK)
    code = text[text.find("\n", idx) + 1:].strip() if idx >= 0 else ""
    return status, reason, code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="적재 없이 미리보기")
    args = ap.parse_args()

    ctx = {p: (elements_context(p), appsettings_context(p)) for p in ("ios", "android")}
    ingested, rejected, pending, skipped = [], [], [], []
    for py in sorted(REVIEW.glob("*_QA-*.py")):
        meta_f = py.with_suffix(".meta.json")
        if not meta_f.exists():
            rejected.append((py.name, "meta 없음")); continue
        status, reason, code = extract(py)
        if status == "SKIP":
            skipped.append((py.stem, reason or "(사유없음)")); continue
        if status != "APPROVED":
            pending.append(py.name); continue
        if not code:
            rejected.append((py.name, "코드 비어있음")); continue
        try:
            ast.parse(code)
        except SyntaxError as e:
            rejected.append((py.name, f"문법오류 line{e.lineno}")); continue
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        key, plat = meta["tc_key"], meta["platform"]
        elem, app = ctx[plat]
        sample = build_sample(meta["tc"], code, plat, elem, app)
        final_lines = len([l for l in code.splitlines() if l.strip()])
        rec = {"tc_key": key, "platform": plat, "tc": meta["tc"], "code": code,
               "training_sample": sample, "source": "bootstrap", "reviewed": True,
               "draft_lines": meta.get("draft_lines"), "final_lines": final_lines}
        if not args.dry:
            (OUT / f"pair_{plat}_{key}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        trim = f"{meta.get('draft_lines')}→{final_lines}줄"
        ingested.append((f"{plat}_{key}", trim))
        print(f"  {'(dry)' if args.dry else '✅'} {plat}/{key}: 적재 ({trim} 트림)")

    # 스킵 사유 로그 누적(재초안 방지·추후 웹뷰 트랙 스코핑용)
    if skipped and not args.dry:
        log = json.loads(SKIPLOG.read_text(encoding="utf-8")) if SKIPLOG.exists() else {}
        log.update({stem: why for stem, why in skipped})
        SKIPLOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n적재 {len(ingested)} | 대기(DRAFT) {len(pending)} | 스킵 {len(skipped)} | 거부 {len(rejected)}")
    for stem, why in skipped:
        print(f"  ⏭️  {stem}: SKIP — {why}")
    for n, why in rejected:
        print(f"  ❌ {n}: {why}")
    if ingested and not args.dry:
        print("다음: python3 prepare_data.py → 재학습 → _validate_6144.py (과생성 전후 비교)")


if __name__ == "__main__":
    main()
