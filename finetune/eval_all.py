#!/usr/bin/env python3
"""
체계적 실사용 품질 평가 — 전체 타깃 페이지를 실제 추론 경로(generate_tc, adapters_best)로
돌려 case 수 / 파싱실패 / 반복루프 / distinct step 을 집계하고 약한 페이지를 찾는다.

- 입력 spec = 하이브리드 OCR 캐시(학습=추론 동일 형식)에서 각 타깃 (pdf,pages) 조립.
- 모델 = adapters_best (현재 v21 batch2). generate_tc를 그대로 호출(충실).
- mlx_lm.load 메모이즈 패치로 모델 1회만 로드(빠름).
- 결과 incremental 저장(eval_all_report.json) → 중단돼도 resume.

사용법: python3 eval_all.py   (백그라운드 권장)
"""
import json, re, sys, time
from pathlib import Path
from collections import Counter

FT = Path(__file__).parent
sys.path.insert(0, str(FT))

# ── 모델 1회 로드: mlx_lm.load 메모이즈 ──
import mlx_lm
_orig_load = mlx_lm.load
_LOADED = {}
def _cached_load(path, adapter_path=None, **kw):
    key = (str(path), str(adapter_path))
    if key not in _LOADED:
        _LOADED[key] = _orig_load(path, adapter_path=adapter_path, **kw)
    return _LOADED[key]
mlx_lm.load = _cached_load

from infer_pipeline import generate_tc, ADAPTER_PATH

_HYB = FT / "data_vlm_ocr/ocr_cache_hybrid.json"
CACHE = json.load(open(_HYB if _HYB.exists() else FT / "data_vlm_ocr/ocr_cache.json", encoding="utf-8"))
REGEN = json.load(open(FT / "data_vlm_ocr/regen_targets_wip.json", encoding="utf-8"))

# temp/best-of 인자. baseline(temp0.35·N1)은 eval_all_report.json, 그 외 별도 리포트.
TEMP = 0.35
if "--temp" in sys.argv:
    TEMP = float(sys.argv[sys.argv.index("--temp") + 1])
BEST_OF = 1
if "--best-of" in sys.argv:
    BEST_OF = int(sys.argv[sys.argv.index("--best-of") + 1])
_suffix = ("" if (TEMP == 0.35 and BEST_OF == 1)
           else f"_t{TEMP}" + (f"_n{BEST_OF}" if BEST_OF > 1 else ""))
REPORT = FT / f"eval_all_report{_suffix}.json"

# valid 셋 식별(held-out 표시용): data_simple/valid.jsonl 의 spec 앞 200자
valid_keys = set()
vf = FT / "data_simple/valid.jsonl"
if vf.exists():
    for l in open(vf, encoding="utf-8"):
        u = json.loads(l)["messages"][1]["content"]
        valid_keys.add(u[u.find("기획서:"):][:200])


def spec_for(aid):
    f = FT / f"raw_data/pair_{aid}.json"
    if not f.exists():
        return None
    sp = json.loads(f.read_text(encoding="utf-8")).get("spec_pdf")
    if not isinstance(sp, dict):
        return None
    name = Path(sp.get("path", "")).name
    pages = sp.get("pages")
    if not pages or not all(f"{name}::{p}" in CACHE for p in pages):
        return None
    return "\n\n".join(CACHE[f"{name}::{p}"] for p in pages)


def analyze(result):
    reqs = result.get("requirements") or result.get("testable_requirements") or []
    if not reqs:
        return {"parse_ok": False, "n_reqs": 0, "total": 0, "distinct": 0, "maxrep": 0}
    keys = []
    for r in reqs:
        for s in r.get("zephyr_tc", {}).get("steps", []):
            if isinstance(s, dict):
                keys.append((s.get("data", "") + "|" + s.get("result", ""))[:120])
            else:  # 드물게 모델이 step을 문자열로 — 견고 처리
                keys.append(str(s)[:120])
    c = Counter(keys)
    return {"parse_ok": True, "n_reqs": len(reqs), "total": len(keys),
            "distinct": len(c), "maxrep": (max(c.values()) if c else 0)}


def main():
    results = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    todo = [a for a in REGEN if a not in results]
    print(f"temp={TEMP} best_of={BEST_OF} | 리포트={REPORT.name} | 평가 대상 {len(REGEN)} / 남음 {len(todo)}")
    for i, aid in enumerate(todo, 1):
        spec = spec_for(aid)
        if not spec:
            results[aid] = {"skip": "no_spec"}
            continue
        t0 = time.time()
        try:
            r = generate_tc(spec, adapter_path=str(ADAPTER_PATH), temp=TEMP, best_of=BEST_OF)
            m = analyze(r)
        except Exception as e:
            m = {"parse_ok": False, "error": str(e)[:120]}
        m["ocr_chars"] = len(spec)
        m["target_cases"] = len(REGEN[aid].get("cases", []))
        uc = "기획서:\n" + spec
        m["split"] = "valid" if uc[uc.find("기획서:"):][:200] in valid_keys else "train"
        m["loop"] = m.get("maxrep", 0) >= 4
        m["sec"] = round(time.time() - t0, 1)
        results[aid] = m
        REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"[{i}/{len(todo)}] {aid}: parse={m.get('parse_ok')} "
              f"req={m.get('n_reqs')} distinct={m.get('distinct')} "
              f"loop={m.get('loop')} ({m['sec']}s)")
    aggregate(results)


def aggregate(results):
    ev = {a: m for a, m in results.items() if "skip" not in m}
    n = len(ev)
    ok = sum(1 for m in ev.values() if m.get("parse_ok"))
    loops = sum(1 for m in ev.values() if m.get("loop"))
    dists = [m.get("distinct", 0) for m in ev.values() if m.get("parse_ok")]
    under = [(a, m) for a, m in ev.items()
             if m.get("parse_ok") and m.get("distinct", 0) < max(1, m.get("target_cases", 0) - 1)]
    weak = [(a, m) for a, m in ev.items()
            if (not m.get("parse_ok")) or m.get("loop") or m.get("distinct", 0) <= 1]
    print("\n" + "=" * 60)
    print(f"평가 완료: {n}개")
    print(f"  파싱 성공: {ok}/{n} ({ok/n*100:.0f}%)")
    print(f"  반복루프 잔존: {loops}/{n} ({loops/n*100:.0f}%)")
    print(f"  distinct step 분포: {dict(sorted(Counter(dists).items()))}")
    print(f"  타깃 대비 부족(under-coverage): {len(under)}개")
    print(f"  약한 페이지(파싱실패/루프/distinct≤1): {len(weak)}개")
    for a, m in weak[:25]:
        print(f"    {a}: parse={m.get('parse_ok')} distinct={m.get('distinct')} "
              f"loop={m.get('loop')} target={m.get('target_cases')} err={m.get('error','')}")
    print("=" * 60)
    REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
