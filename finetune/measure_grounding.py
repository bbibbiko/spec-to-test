#!/usr/bin/env python3
"""
grounding 측정 — 학습 데이터의 입력(OCR spec)이 출력(target case) 내용을 얼마나 담는가.

초기 측정 결과 중앙값 28%, 40% 미만이 74%였다 → 모델이 입력에 없는 내용을 생성하도록
강요받아 환각·반복이 발생. OCR 보강 후 이 수치가 개선되는지로 효과를 정량 검증한다.

사용법:
  python3 measure_grounding.py                      # data_simple/train.jsonl
  python3 measure_grounding.py --data data_simple/valid.jsonl
"""
import argparse
import json
import re
import statistics


def words(t: str) -> set:
    return set(re.findall(r"[가-힣a-zA-Z]{2,}", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_simple/train.jsonl")
    args = ap.parse_args()

    overlaps = []
    for line in open(args.data, encoding="utf-8"):
        d = json.loads(line)
        tgt = json.loads(d["messages"][2]["content"])
        cases = tgt.get("cases") or (
            [c for s in tgt.get("screens", []) for c in s.get("cases", [])])
        if not cases:
            continue
        uc = d["messages"][1]["content"]
        spec = uc[uc.find("기획서:") + 4:] if "기획서:" in uc else uc
        spec_w = words(spec)
        tgt_w = set()
        for c in cases:
            tgt_w |= words(c.get("check", "") + " " + c.get("expect", "") + " " + c.get("cond", ""))
        if not tgt_w:
            continue
        overlaps.append(len(tgt_w & spec_w) / len(tgt_w))

    n = len(overlaps)
    print(f"데이터: {args.data} ({n}개 샘플)")
    print(f"target→spec grounding: 중앙 {statistics.median(overlaps)*100:.0f}%, "
          f"평균 {statistics.mean(overlaps)*100:.0f}%")
    buckets = {"<40%": 0, "40-60%": 0, "60-80%": 0, ">=80%": 0}
    for o in overlaps:
        if o < 0.4: buckets["<40%"] += 1
        elif o < 0.6: buckets["40-60%"] += 1
        elif o < 0.8: buckets["60-80%"] += 1
        else: buckets[">=80%"] += 1
    print("분포:", {k: v for k, v in buckets.items()})
    print(f"기준선(OCR 보강 전): 중앙 28%, <40% 74%")


if __name__ == "__main__":
    main()
