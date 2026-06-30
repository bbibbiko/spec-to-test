#!/usr/bin/env python3
"""
코드 제너레이터 학습 데이터 전처리
raw_data/pair_*.json → train.jsonl / valid.jsonl

사용법:
  python3 prepare_data.py --input-dir ./raw_data --output-dir ./data
"""

import json
import random
import argparse
from pathlib import Path


def main():
    _here = Path(__file__).parent  # 실행 위치 무관
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(_here / "raw_data"))
    parser.add_argument("--output-dir", default=str(_here / "data"))
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    # 새 포맷(pair_{platform}_QA-*.json)만 — 옛 iOS-only(pair_QA-*.json) 제외
    files = sorted(input_dir.glob("pair_ios_*.json")) + \
        sorted(input_dir.glob("pair_android_*.json"))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if "training_sample" not in data:
            print(f"  ⚠️  {f.name}: training_sample 없음 (건너뜀)")
            continue
        samples.append(data["training_sample"])
        print(f"  ✅ {f.name}")

    if not samples:
        print("❌ 샘플이 없습니다.")
        return

    random.shuffle(samples)
    n_valid = max(1, int(len(samples) * args.valid_ratio))
    valid_samples = samples[:n_valid]
    train_samples = samples[n_valid:]

    for name, data in [("train", train_samples), ("valid", valid_samples)]:
        path = output_dir / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in data) + "\n",
            encoding="utf-8"
        )
        print(f"  💾 {name}.jsonl: {len(data)}개")

    print(f"\n✅ 완료: train {len(train_samples)}개 / valid {len(valid_samples)}개")


if __name__ == "__main__":
    main()
