#!/usr/bin/env python3
"""부트스트랩 ①초안 생성 — 코드 없는 TC를 모델로 초안 코드화 → 사람 리뷰 대기.

흐름: 후보 TC → mlx_code_generator 초안 → bootstrap/review/
에 편집용 .py + .meta.json 저장. 사람이 .py를 수정하고 STATUS를 APPROVED로 바꾸면
bootstrap_ingest.py가 학습셋(raw_data/pair_*.json)으로 적재.

사용법:
  ZEPHYR_API_TOKEN=<PAT> python3 bootstrap_draft.py --keys 35530,35603 --platform both
  ZEPHYR_API_TOKEN=<PAT> python3 bootstrap_draft.py --auto 5      # 코드없는 TC 자동선정 N개
"""
import os, re, sys, json, argparse, time
from pathlib import Path

FT = Path(__file__).parent
sys.path.insert(0, str(FT))
sys.path.insert(0, str(FT.parent))  # jira_client
REVIEW = FT / "bootstrap" / "review"
TESTS = FT.parent.parent.parent / "tests"
TCGEN_RAW = FT.parent / "finetune" / "raw_data"   # TC-gen 타깃(키 풀)

from collect_data import (fetch_tc, tc_to_training_format, elements_context,
                          appsettings_context, load_test_files)


def coded_keys() -> set:
    """이미 코드(테스트 파일)가 있는 QA 키 — 후보에서 제외."""
    s = set()
    for plat in ("ios", "android"):
        s |= {k for k in load_test_files(plat)}  # 'QA-12345'
    return s


def seen_keys() -> set:
    """이미 초안 생성됐거나(review/) 스킵된(skipped.json) 키 — 재초안 방지."""
    s = set()
    for f in REVIEW.glob("*_QA-*.py"):
        m = re.search(r"(QA-\d+)", f.name)
        if m:
            s.add(m.group(1))
    skiplog = REVIEW.parent / "skipped.json"
    if skiplog.exists():
        import json as _j
        for stem in _j.loads(skiplog.read_text(encoding="utf-8")):
            m = re.search(r"(QA-\d+)", stem)
            if m:
                s.add(m.group(1))
    return s


def candidate_keys(n: int) -> list:
    """TC-gen 타깃 키 중 아직 코드·초안·스킵 안 된 것 N개(다양성 위해 정렬만)."""
    exclude = coded_keys() | seen_keys()
    pool = []
    for f in sorted(TCGEN_RAW.glob("pair_QA-*.json")):
        m = re.search(r"(QA-\d+)", f.name)
        if m and m.group(1) not in exclude:
            pool.append(m.group(1))
    return pool[:n]


def step_comment(tc_dict: dict) -> str:
    out = [f"# --- TC: {' '.join((tc_dict.get('summary','') or '').split())[:80]} ---"]
    for s in tc_dict.get("steps", [])[:12]:
        txt = " ".join((s.get("step") or "").split())[:80]  # 줄바꿈 제거(주석 깨짐 방지)
        out.append(f"#  step{s.get('index')}: {txt}")
    return "\n".join(out)


def write_review(key: str, plat: str, tc_dict: dict, code: str):
    REVIEW.mkdir(parents=True, exist_ok=True)
    n_draft = len([l for l in code.splitlines() if l.strip()])
    py = REVIEW / f"{plat}_{key}.py"
    py.write_text(
        f"# === REVIEW | {key} | {plat} | STATUS: DRAFT ===\n"
        f"# STATUS → APPROVED=적재 / SKIP (사유)=제외(웹뷰 등 검증불가) / DRAFT=대기. "
        f"아래 코드 검수·수정(과생성 트림).\n"
        f"{step_comment(tc_dict)}\n"
        f"# ===== 코드 (이 줄 아래만 학습에 사용) =====\n"
        f"{code}\n", encoding="utf-8")
    (REVIEW / f"{plat}_{key}.meta.json").write_text(
        json.dumps({"tc_key": key, "platform": plat, "tc": tc_dict,
                    "draft_lines": n_draft}, ensure_ascii=False, indent=1), encoding="utf-8")
    return n_draft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="", help="QA-XXXXX 콤마구분(접두사 없이 숫자만도 허용)")
    ap.add_argument("--auto", type=int, default=0, help="코드없는 TC 자동선정 N개")
    ap.add_argument("--platform", default="both", choices=["both", "ios", "android"])
    ap.add_argument("--jira-url", default=os.getenv("JIRA_URL", "https://jira.example.com"))
    ap.add_argument("--jira-token", default=os.getenv("ZEPHYR_API_TOKEN"))
    ap.add_argument("--delay", type=float, default=0.7)
    args = ap.parse_args()
    if not args.jira_token:
        print("❌ ZEPHYR_API_TOKEN 필요"); sys.exit(1)

    keys = []
    for k in args.keys.split(",") if args.keys else []:
        k = k.strip()
        if k:
            keys.append(k if k.startswith("QA-") else f"QA-{k}")
    if args.auto:
        keys += candidate_keys(args.auto)
    keys = list(dict.fromkeys(keys))
    if not keys:
        print("❌ --keys 또는 --auto 필요"); sys.exit(1)
    plats = ["ios", "android"] if args.platform == "both" else [args.platform]
    print(f"후보 {len(keys)}키 × {plats} → 초안 생성")

    from jira_client import JiraClient
    from mlx_code_generator import MLXCodeGenerator
    client = JiraClient(base_url=args.jira_url, api_token=args.jira_token)
    gen = MLXCodeGenerator()
    ctx = {p: (elements_context(p), appsettings_context(p)) for p in plats}

    made, skip = 0, []
    for key in keys:
        tc, err = fetch_tc(client, key)
        if not tc:
            skip.append((key, err)); print(f"  ❌ {key}: {err}"); time.sleep(args.delay); continue
        tc_dict = tc_to_training_format(tc)
        for p in plats:
            elem, app = ctx[p]
            code = gen.generate(tc_dict, platform=p)  # 초안
            n = write_review(key, p, tc_dict, code)
            made += 1
            print(f"  ✅ {key}/{p}: 초안 {n}줄 → bootstrap/review/{p}_{key}.py")
        time.sleep(args.delay)

    print(f"\n초안 {made}개 생성 | fetch실패 {len(skip)}")
    print(f"다음: bootstrap/review/*.py 검수·수정 → STATUS APPROVED → python3 bootstrap_ingest.py")


if __name__ == "__main__":
    main()
