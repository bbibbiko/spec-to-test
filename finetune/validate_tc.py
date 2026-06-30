#!/usr/bin/env python3
"""
TC step 열거 검증 하베스트 (메트릭 = val_loss 아님, 실제 추론 step 수/중복)

목표: 1페이지 → 1 TC + 페이지의 각 케이스가 하나의 step.
성공 기준은 "req 여러 개"가 아니라 "1 TC 안에 케이스 수만큼 distinct step"이 나오는지.
실패 모드는 케이스별로 step을 나누지 못하고 한 step을 무한 반복하는 경우.

train/valid 샘플의 spec(입력)을 그대로 모델에 투입해:
  - 생성된 총 step 수 / 고유(distinct) step 수
  - 정답 step 수 대비 커버리지
  - 반복 루프 잔존 여부
를 정량 측정한다. repetition_penalty / 재학습이 이 증상을 고치는지 비교용.

사용법:
  python3 validate_tc.py --train data_vlm_ocr/valid.jsonl --min-steps 3   # held-out
  python3 validate_tc.py --train data_vlm_ocr/train.jsonl --min-steps 4   # floor
  python3 validate_tc.py --repetition-penalty 1.0                          # penalty 끄고 비교
  python3 validate_tc.py --spec spec_sample_multiagent_p20.txt --expected-steps 6
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from infer_pipeline import generate_tc, ADAPTER_PATH

SPEC_MARKER = "기획서:\n"


def target_steps(sample: dict) -> int:
    """학습 정답의 총 step 수."""
    try:
        reqs = json.loads(sample["messages"][2]["content"]).get("requirements", [])
        return sum(len(r.get("zephyr_tc", {}).get("steps", [])) for r in reqs)
    except Exception:
        return 0


def extract_spec(user_content: str) -> str:
    i = user_content.find(SPEC_MARKER)
    return user_content[i + len(SPEC_MARKER):] if i >= 0 else user_content


def analyze_output(result: dict) -> tuple[int, int, int]:
    """(총 step, 고유 step, 최다 반복 step 횟수) 반환.

    구분 내용은 data/result에 있으므로 step+data+result 전체로 distinct 판정
    (step 필드는 [Ver.] boilerplate라 거의 동일 — 이것만 보면 오측정).
    """
    steps = []
    for r in result.get("requirements", []):
        for s in r.get("zephyr_tc", {}).get("steps", []):
            if isinstance(s, dict):
                txt = s.get("step", "") + "|" + s.get("data", "") + "|" + s.get("result", "")
            else:
                txt = str(s)
            steps.append(" ".join(txt.split())[:200])
    if not steps:
        return 0, 0, 0
    c = Counter(steps)
    return len(steps), len(c), max(c.values())


def run_one(spec, adapter, max_tokens, rep, expected):
    result = generate_tc(spec, adapter, max_tokens, repetition_penalty=rep)
    total, distinct, maxrep = analyze_output(result)
    n_req = len(result.get("requirements", []))
    loop = maxrep >= 4
    ok = distinct >= max(2, min(expected, 3)) and not loop
    print(f"    → req {n_req}, step 총{total}/고유{distinct} (최다반복 {maxrep}회), "
          f"정답 step {expected}  {'✅' if ok else '❌'}{' [반복루프]' if loop else ''}\n")
    return distinct, total, loop, ok


def main():
    ap = argparse.ArgumentParser(description="TC step 열거 검증")
    ap.add_argument("--train", default="data_vlm_ocr/valid.jsonl")
    ap.add_argument("--spec", default=None, help="단일 spec 파일 직접 테스트")
    ap.add_argument("--expected-steps", type=int, default=6, help="--spec 모드 정답 step 수")
    ap.add_argument("--adapter", default=str(ADAPTER_PATH))
    ap.add_argument("--min-steps", type=int, default=3, help="정답 step 수 하한(이 이상만 검증)")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    args = ap.parse_args()

    print(f"\nadapter={args.adapter}, rep_penalty={args.repetition_penalty}")

    if args.spec:
        spec = Path(args.spec).read_text(encoding="utf-8")
        print(f"\n[단일] {args.spec} ({len(spec)}자), 정답 step {args.expected_steps}")
        run_one(spec, args.adapter, args.max_tokens, args.repetition_penalty, args.expected_steps)
        return

    samples = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    seen = set()
    targets = []
    for s in samples:
        exp = target_steps(s)
        if exp < args.min_steps:
            continue
        spec = extract_spec(s["messages"][1]["content"])
        if spec[:200] in seen:
            continue
        seen.add(spec[:200])
        targets.append((spec, exp))
    targets = targets[:args.limit]

    print(f"검증 대상: step≥{args.min_steps} 고유 샘플 {len(targets)}개 ({args.train})\n")
    rows = []
    for i, (spec, exp) in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] 정답 step {exp} — spec {len(spec)}자")
        rows.append(run_one(spec, args.adapter, args.max_tokens, args.repetition_penalty, exp))

    ok = sum(1 for *_, o in rows if o)
    loops = sum(1 for *_, l, _ in rows if l)
    print("=" * 55)
    print(f"요약: 충분한 step 생성 {ok}/{len(rows)}, 반복루프 잔존 {loops}/{len(rows)}")
    print(f"  고유 step 수: {[d for d, _, _, _ in rows]}")
    print("=" * 55)


if __name__ == "__main__":
    main()
