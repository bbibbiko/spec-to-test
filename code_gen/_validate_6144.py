#!/usr/bin/env python3
"""code_gen 어댑터 품질 검증 (올바른 지표).

지표 정정: 기존 max_line_repeat(전체 등장 횟수)는 `except Exception as e:` 같은
방어적 코딩의 구조적 반복을 루프로 오탐했다(정답 코드에 except가 20회 이상 등장하는 경우 등).
degenerate 루프는 연속(back-to-back) 반복이어야 하므로 다음으로 교체했다:
  - max_consec_line: 연속 동일 비자명 라인 최대 런
  - max_block: 연속 반복되는 2~3줄 블록 최대 횟수
  - py_valid: ast.parse 통과 (가장 중요한 품질 게이트)
  - gen/ref: 과생성(>1.5)·저생성(<0.6) 비율
판정: loop = (연속런≥4 or 블록≥3). temp 0.1 + rep_penalty 1.1.
"""
import json, re, sys, ast
from pathlib import Path

FT = Path(__file__).parent
sys.path.insert(0, str(FT))
sys.path.insert(0, str(FT.parent))

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

ADAPTER = str(FT / "adapters")
MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
LONG = {"35528", "33640", "43692", "40856", "41446"}
REPORT = FT / "validate_report.json"


def max_consec_line(lines: list) -> int:
    mx = run = 1
    prev = None
    for l in lines:
        if l == prev and len(l) > 5:
            run += 1
            mx = max(mx, run)
        else:
            run = 1
            prev = l
    return mx


def max_block(lines: list, win: int = 2) -> int:
    """연속 반복되는 win줄 블록 최대 횟수 (degenerate 블록 루프 탐지)."""
    mx, i = 1, 0
    n = len(lines)
    while i < n - win:
        block = tuple(lines[i:i + win])
        if len("".join(block).strip()) <= 8:
            i += 1
            continue
        cnt, j = 1, i + win
        while tuple(lines[j:j + win]) == block:
            cnt += 1
            j += win
        mx = max(mx, cnt)
        i = j if cnt > 1 else i + 1
    return mx


def analyze(code: str, exp_lines: int) -> dict:
    lines = [l.strip() for l in code.splitlines() if l.strip()]
    try:
        ast.parse(code)
        ok = True
    except Exception:
        ok = False
    consec = max_consec_line(lines)
    block = max_block(lines, 2)
    loop = consec >= 4 or block >= 3
    ratio = len(lines) / max(exp_lines, 1)
    if loop:
        verdict = "loop"
    elif not ok:
        verdict = "invalid"
    elif ratio > 1.5:
        verdict = "over"
    elif ratio < 0.6:
        verdict = "under"
    else:
        verdict = "ok"
    return {"py": ok, "consec": consec, "block": block, "loop": loop,
            "gen": len(lines), "ref": exp_lines, "ratio": round(ratio, 2),
            "verdict": verdict}


def main():
    print("🔄 어댑터 로드...")
    model, tok = load(MODEL_ID, adapter_path=ADAPTER)
    print("✅ 로드 완료\n")
    sampler = make_sampler(temp=0.1)
    lp = make_logits_processors(repetition_penalty=1.1)

    results = {}
    for f in sorted(FT.glob("raw_data/pair_*.json")):
        key = re.search(r"QA-(\d+)", f.name).group(1)
        plat = "ios" if "ios" in f.name else "android"
        d = json.loads(f.read_text())
        msgs = d["training_sample"]["messages"]
        exp_lines = len([l for l in msgs[2]["content"].splitlines() if l.strip()])
        prompt = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
        out = generate(model, tok, prompt=prompt, max_tokens=2500, verbose=False,
                       sampler=sampler, logits_processors=lp)
        out = out.split("<|im_end|>")[0]
        mm = re.search(r"```python\s*(.*?)(?:```|$)", out, re.DOTALL)
        code = (mm.group(1) if mm else out).strip()
        m = analyze(code, exp_lines)
        m["long"] = key in LONG
        results[f"{key}_{plat}"] = m
        tag = "장문" if m["long"] else "    "
        print(f"  {tag} {key} {plat}: py={m['py']} 연속{m['consec']} 블록{m['block']} "
              f"{m['gen']}/{m['ref']}줄(x{m['ratio']}) → {m['verdict']}")
        REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=1))

    print("\n" + "=" * 60)
    from collections import Counter
    vc = Counter(m["verdict"] for m in results.values())
    pyok = sum(1 for m in results.values() if m["py"])
    loops = sum(1 for m in results.values() if m["loop"])
    print(f"전체 {len(results)}개 | py유효 {pyok} | 실제 루프 {loops} | 판정분포 {dict(vc)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
