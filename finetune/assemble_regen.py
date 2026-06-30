#!/usr/bin/env python3
"""
재생성 타겟(regen_targets_wip.json) + OCR 캐시 → data_simple/{train,valid}.jsonl 조립.

grounding 개선: 사람이 작성한 TC 대신 OCR에서 도출한 타깃으로 학습 → 입력·출력 정합.
- user = infer_pipeline simple_mode와 동일 템플릿 + 해당 (pdf,pages) OCR spec
- assistant = regen 타겟(simple format) 그대로
- system = SIMPLE_SYSTEM_PROMPT

사용법: python3 assemble_regen.py   # data_simple/{train,valid}.jsonl 생성
"""
import json
import random
from pathlib import Path
from simple_format import SIMPLE_SYSTEM_PROMPT

FT = Path(__file__).parent
# 하이브리드 OCR 캐시가 있으면 우선 사용(텍스트 PDF=pymupdf, 학습=추론 일치).
# 없으면 기존 VLM 캐시. (rebuild_ocr_cache_hybrid.py 로 생성)
_HYBRID = FT / "data_vlm_ocr/ocr_cache_hybrid.json"
_CACHE_PATH = _HYBRID if _HYBRID.exists() else FT / "data_vlm_ocr/ocr_cache.json"
print(f"📂 OCR 캐시: {_CACHE_PATH.name}")
CACHE = json.load(open(_CACHE_PATH, encoding="utf-8"))
REGEN = json.load(open(FT / "data_vlm_ocr/regen_targets_wip.json", encoding="utf-8"))

USER_TMPL = (
    "다음 기획서에서 테스트케이스를 작성해주세요.\n\n"
    "주의사항:\n"
    "- 기획서에 정의된 모든 항목을 누락 없이 포함하세요.\n"
    "- 기획서 내용으로 예상 가능한 예외 케이스(빈 상태, 긴 텍스트, 오류 상황 등)도 포함하세요.\n\n"
    "기획서:\n{spec}"
)


def spec_for(aid: str) -> str | None:
    """raw_data/pair_{aid}.json 의 (pdf,pages) → OCR 캐시 join 으로 spec 재구성."""
    f = FT / f"raw_data/pair_{aid}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    sp = d.get("spec_pdf")
    if not isinstance(sp, dict):
        return None
    name = Path(sp.get("path", "")).name
    pages = sp.get("pages")
    if not pages:
        return None
    parts = []
    for p in pages:
        key = f"{name}::{p}"
        if key not in CACHE:
            return None  # OCR 미완료 페이지 포함 → 스킵
        parts.append(CACHE[key])
    return "\n\n".join(parts)


def main():
    samples = []
    skipped = []
    for aid, target in REGEN.items():
        spec = spec_for(aid)
        if not spec:
            skipped.append(aid)
            continue
        samples.append({"messages": [
            {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
            {"role": "user", "content": USER_TMPL.format(spec=spec)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, indent=2)},
        ]})

    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, round(len(samples) * 0.15))
    valid, train = samples[:n_val], samples[n_val:]

    out = FT / "data_simple"
    out.mkdir(exist_ok=True)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in train) + "\n", encoding="utf-8")
    (out / "valid.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in valid) + "\n", encoding="utf-8")
    print(f"조립: train {len(train)} / valid {len(valid)} (전체 {len(samples)})")
    if skipped:
        print(f"스킵(OCR 미완 or raw_data 없음) {len(skipped)}개: {skipped[:8]}")


if __name__ == "__main__":
    main()
