#!/usr/bin/env python3
"""
코드 제너레이터 학습 데이터 수집 (단일 모델 + 플랫폼 조건부)

tests/{ios,android}/test_qa_*.py ↔ Zephyr TC 를 매핑해
(TC + 플랫폼 + elements/app_settings 컨텍스트 → 코드) 학습 쌍을 만든다.

설계: iOS/Android를 분리 학습하지 않고 한 모델을 플랫폼 플래그로 조건화한다.
코드 차이가 대부분 로케이터(elements.py)와 소수 관용구라, 로케이터는 컨텍스트로 주입하고
플랫폼만 플래그로 구분 → 작은 데이터를 사실상 2배로 활용.

사용법:
  ZEPHYR_API_TOKEN=... python3 collect_data.py            # ios+android 둘 다
  ZEPHYR_API_TOKEN=... python3 collect_data.py --platform ios
  → raw_data/pair_{platform}_{key}.json 생성 → prepare_data.py 가 병합
"""

import json
import re
import os
import sys
import time
import argparse
from pathlib import Path

FINETUNE_DIR = Path(__file__).parent.parent          # tools/tc-generator
TESTS_DIR    = FINETUNE_DIR.parent.parent / "tests"  # repo/tests
TC_GEN_DIR   = FINETUNE_DIR
OUTPUT_DIR   = Path(__file__).parent / "raw_data"

sys.path.insert(0, str(TC_GEN_DIR))

# 플랫폼 무관 — 단일 모델이 플랫폼 플래그로 ios/android 코드를 구분 생성
SYSTEM_PROMPT = """당신은 모바일 앱 테스트 자동화 전문가입니다.
Zephyr 테스트케이스(TC)를 Python Appium 테스트 코드로 변환합니다.

코드 작성 규칙:
- 함수명: test_qa_{tc_id} 형식
- step_reporter.step(n, "설명") 컨텍스트 매니저로 스텝 구분
- WebDriverWait + expected_conditions(EC) 사용
- 주어진 elements.py 로케이터 클래스와 app_settings.py 유틸 함수만 사용
- assert 문으로 검증
- 플랫폼(ios/android)에 맞는 관용구 사용 (예: android는 driver.back() 가능)

반드시 ```python 코드블록의 완전한 함수 하나만 응답하세요."""

# 컨텍스트 토큰 예산(문자수). max_seq_length 3072 내에서 TC+코드 자리 확보.
ELEM_BUDGET = 1800
APP_BUDGET = 1100


def _cap(text: str, budget: int) -> str:
    return text if len(text) <= budget else text[:budget].rsplit("\n", 1)[0] + "\n…(생략)"


def elements_context(platform: str) -> str:
    """elements.py → 'Class: ATTR1, ATTR2 …' 압축(로케이터 이름만). By 상수클래스 제외."""
    f = TESTS_DIR / platform / "elements.py"
    if not f.exists():
        return ""
    groups: dict[str, list] = {}
    cur = None
    for line in f.read_text(encoding="utf-8").splitlines():
        cm = re.match(r"class (\w+)", line)
        if cm:
            cur = cm.group(1)
            if cur != "By":
                groups.setdefault(cur, [])
            else:
                cur = None
            continue
        am = re.match(r"\s+([A-Z][A-Z0-9_]+)\s*=", line)
        if cur and am:
            groups[cur].append(am.group(1))
    lines = [f"{c}: {', '.join(a)}" for c, a in groups.items() if a]
    return _cap("\n".join(lines), ELEM_BUDGET)


def appsettings_context(platform: str) -> str:
    """app_settings.py → 사용 가능한 유틸 함수 시그니처 목록."""
    f = TESTS_DIR / platform / "app_settings.py"
    if not f.exists():
        return ""
    sigs = re.findall(r"^def (\w+\([^)]*\))", f.read_text(encoding="utf-8"), re.MULTILINE)
    return _cap("\n".join(sigs), APP_BUDGET)


def load_test_files(platform: str) -> dict:
    """test_qa_*.py → {QA-키: 함수코드}."""
    test_dir = TESTS_DIR / platform
    result = {}
    for f in sorted(test_dir.glob("test_qa_*.py")):
        m = re.search(r"test_qa_(\d+)", f.name)
        if not m:
            continue
        code = f.read_text(encoding="utf-8").strip()
        func_m = re.search(r"(def test_qa_\d+\(.*?\):.*?)(?=\ndef |\Z)", code, re.DOTALL)
        result[f"QA-{m.group(1)}"] = func_m.group(1).strip() if func_m else code
    return result


def tc_to_training_format(tc) -> dict:
    return {
        "id": tc.key,
        "summary": tc.name or tc.key,
        "precondition": tc.precondition or "",
        "steps": [{"index": s.index, "step": s.description,
                   "data": s.test_data, "expected": s.expected_result}
                  for s in tc.steps],
    }


def build_user_prompt(tc_dict: dict, platform: str, elem_ctx: str, app_ctx: str) -> str:
    """학습=추론 공용 user 프롬프트. 추론기(mlx_code_generator)도 이걸 import해서 사용."""
    return (
        f"다음 Zephyr TC를 Python Appium 테스트 코드로 변환해주세요.\n\n"
        f"플랫폼: {platform}\n\n"
        f"[사용 가능한 로케이터 — elements.py]\n{elem_ctx}\n\n"
        f"[사용 가능한 유틸 — app_settings.py]\n{app_ctx}\n\n"
        f"[TC]\n{json.dumps(tc_dict, ensure_ascii=False, indent=2)}"
    )


def build_sample(tc_dict: dict, func_code: str, platform: str,
                 elem_ctx: str, app_ctx: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(tc_dict, platform, elem_ctx, app_ctx)},
        {"role": "assistant", "content": f"```python\n{func_code}\n```"},
    ]}


def fetch_tc(client, key: str, tries: int = 5):
    """TC fetch — 429(rate limit)·0스텝(throttle 의심) 시 백오프 재시도.
    TC는 키당 1개라 ios/android 공용 → 캐시로 중복 fetch 방지."""
    for i in range(tries):
        try:
            tc = client.get_test_case(key)
            if tc.steps:                      # 정상(스텝 있음)
                return tc, None
            err = "0스텝(throttle 의심)"      # 스텝 0 = 보통 throttle → 재시도
        except Exception as e:
            err = str(e)[:80]
        if i < tries - 1:
            time.sleep(2.0 * (i + 1))          # 2,4,6,8초 백오프
    return None, err


def main():
    parser = argparse.ArgumentParser(description="코드 생성 학습 데이터 수집(단일모델)")
    parser.add_argument("--jira-url", default=os.getenv("JIRA_URL", "https://jira.example.com"))
    parser.add_argument("--jira-token", default=os.getenv("ZEPHYR_API_TOKEN"))
    parser.add_argument("--platform", default="both", choices=["both", "ios", "android"])
    parser.add_argument("--delay", type=float, default=0.7, help="요청 간 간격(초, rate limit 완화)")
    args = parser.parse_args()

    if not args.jira_token:
        print("❌ --jira-token 또는 ZEPHYR_API_TOKEN 필요")
        sys.exit(1)

    from jira_client import JiraClient
    client = JiraClient(base_url=args.jira_url, api_token=args.jira_token)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    platforms = ["ios", "android"] if args.platform == "both" else [args.platform]
    code = {p: load_test_files(p) for p in platforms}      # platform → {key: code}
    ctx = {p: (elements_context(p), appsettings_context(p)) for p in platforms}
    for p in platforms:
        print(f"📂 [{p}] 코드 {len(code[p])}개 "
              f"(elements {len(ctx[p][0])}자 / app_settings {len(ctx[p][1])}자)")

    # TC는 키당 1회만 fetch(캐시) — 중복 제거로 rate limit·시간 절감
    all_keys = sorted({k for p in platforms for k in code[p]})
    print(f"\n🔻 고유 TC {len(all_keys)}개 fetch (키당 1회, 간격 {args.delay}s)...")
    tc_cache, tc_fail = {}, {}
    for key in all_keys:
        tc, err = fetch_tc(client, key)
        if tc:
            tc_cache[key] = tc_to_training_format(tc)
            print(f"  ✅ {key} ({len(tc_cache[key]['steps'])}스텝)")
        else:
            tc_fail[key] = err
            print(f"  ❌ {key}: {err}")
        time.sleep(args.delay)

    # 캐시된 TC + 각 플랫폼 코드로 쌍 생성
    total_ok, total_skip = [], []
    for p in platforms:
        elem_ctx, app_ctx = ctx[p]
        for key, func_code in code[p].items():
            if key not in tc_cache:
                total_skip.append((p, key)); continue
            sample = build_sample(tc_cache[key], func_code, p, elem_ctx, app_ctx)
            (OUTPUT_DIR / f"pair_{p}_{key}.json").write_text(json.dumps({
                "tc_key": key, "platform": p,
                "tc": tc_cache[key], "code": func_code, "training_sample": sample,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            total_ok.append((p, key))

    print(f"\n{'='*56}")
    print(f"✅ 수집 {len(total_ok)}쌍 / TC fetch 실패 {len(tc_fail)}키 / 쌍 스킵 {len(total_skip)}")
    if tc_fail:
        print(f"   fetch 실패 키: {list(tc_fail)}")
    print(f"📁 {OUTPUT_DIR}")
    print(f"다음: python3 prepare_data.py  → data/{{train,valid}}.jsonl 병합")


if __name__ == "__main__":
    main()
