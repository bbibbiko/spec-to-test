#!/usr/bin/env python3
"""
data_vlm_ocr/{train,valid}.jsonl (full Zephyr 타겟) → 단순포맷 학습데이터 생성.

Phase4: 모델 타겟을 {id,description,entry,cases:[{cond,check,expect}]}로 단순화
(simple_format.zephyr_to_simple). 추론은 infer_pipeline simple_mode가 조립.

사용법(집에서 OCR로 train/valid 재생성 후):
  python3 make_simple_data.py
  → data_simple/{train,valid}.jsonl 생성 (config의 data: data_simple로 학습)
"""
import json
from pathlib import Path
from simple_format import zephyr_to_simple, SIMPLE_SYSTEM_PROMPT

SRC = Path("data_vlm_ocr")
DST = Path("data_simple")


def convert(reqs: list) -> dict:
    simples = [zephyr_to_simple(r) for r in reqs]
    return simples[0] if len(simples) == 1 else {"screens": simples}


def main():
    DST.mkdir(exist_ok=True)
    for fn in ["train", "valid"]:
        out = []
        for line in open(SRC / f"{fn}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            try:
                reqs = json.loads(d["messages"][2]["content"]).get("requirements", [])
            except Exception:
                continue
            if not reqs:
                continue
            target = convert(reqs)
            out.append(json.dumps({"messages": [
                {"role": "system", "content": SIMPLE_SYSTEM_PROMPT},
                d["messages"][1],
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, indent=2)},
            ]}, ensure_ascii=False))
        (DST / f"{fn}.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"{fn}: {len(out)}개 → {DST}/{fn}.jsonl")


if __name__ == "__main__":
    main()
