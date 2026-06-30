#!/usr/bin/env python3
"""
Phase 4: 출력 단순화 — 모델은 짧은 {check, expect} 케이스 리스트만 생성,
결정적 코드가 Zephyr TC 포맷으로 조립한다.

복잡한 중첩 Zephyr JSON(steps × {data,result} 멀티라인)을 모델이 직접 만들면
자유텍스트 필드에서 폭주(반복 붕괴)한다(진단 확인). 과제를 단순화해 7.8B로도
안정 생성 + grounding 개선 + 추후 더 작고 빠른 모델까지 가능케 한다.

- zephyr_to_simple(req): 기존 학습 타겟 → 단순 포맷 (학습 데이터 재가공용)
- simple_to_zephyr(simple, pdf, page): 단순 포맷 → Zephyr TC (추론 후 조립용)
"""

import re

# 단순포맷 학습/추론 공통 시스템 프롬프트 (prepare=infer 일치)
SIMPLE_SYSTEM_PROMPT = """당신은 모바일 앱 테스트케이스 작성 전문가입니다.
기획서에서 확인해야 할 항목을 빠짐없이 추출하여 JSON으로 반환합니다.

출력 형식:
{
  "id": "화면 ID",
  "category": "카테고리",
  "description": "화면명",
  "cases": [
    {"cond": "조건(있을 때만)", "check": "확인 동작", "expect": "기대 결과"}
  ]
}

규칙:
- 화면의 각 UI 요소·동작·상태·예외를 하나의 case로 만든다.
- check는 짧은 확인 동작 한 줄, expect는 기대 결과를 간결히.
- 동일 동작이 조건(OS, 상태 등)에 따라 다르면 cond로 구분한다.
- 같은 case를 반복하지 말 것. 반드시 유효한 JSON만, 모든 값은 한글로."""

MARKER_RE = re.compile(r"\*\[[^\]]+\]\*")
NUM_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def _strip_marker_lines(text: str) -> list[str]:
    """마커(*[Step]* 등) 제거하고 비어있지 않은 줄 리스트 반환 (번호접두사 유지)."""
    body = MARKER_RE.sub("", text)
    return [ln.strip() for ln in body.split("\n") if ln.strip()]


def _join_clean(lines: list[str]) -> str:
    """번호접두사 제거하고 ' / '로 합침."""
    return " / ".join(NUM_PREFIX_RE.sub("", ln).strip() for ln in lines)


def _extract_precondition(step_field: str) -> str:
    """step 필드의 *[Precondition]* 내용 추출 (없으면 '')."""
    m = re.search(r"\*\[Precondition\]\*(.*)", step_field, re.DOTALL)
    if not m:
        return ""
    return _join_clean([ln.strip() for ln in m.group(1).split("\n") if ln.strip()])


def zephyr_to_simple(req: dict) -> dict:
    """기존 Zephyr req → 단순 포맷 (완화책: 네비 entry 보존 + expect 구조 보존).

    - entry(화면 레벨): 첫 step 네비게이션(data 마지막 줄 제외) — '어떻게 화면 진입'.
    - 각 case: cond(precondition) + check(data 마지막 줄, 확인 동작)
               + expect(result 내용을 번호/계층 그대로 보존, 평탄화 안 함).
    """
    steps = req.get("zephyr_tc", {}).get("steps", [])
    entry: list[str] = []
    for s in steps:
        dl = _strip_marker_lines(s.get("data", ""))
        if len(dl) > 1:
            entry = [NUM_PREFIX_RE.sub("", x).strip() for x in dl[:-1]]
            break

    cases = []
    for s in steps:
        data_lines = _strip_marker_lines(s.get("data", ""))
        result_lines = _strip_marker_lines(s.get("result", ""))
        cond = _extract_precondition(s.get("step", ""))
        check = NUM_PREFIX_RE.sub("", data_lines[-1]).strip() if data_lines else ""
        expect = "\n".join(result_lines)   # 계층/번호 원형 보존
        if check or expect or cond:
            case = {"check": check, "expect": expect}
            if cond:
                case = {"cond": cond, **case}
            cases.append(case)
    out = {
        "id": req.get("id", ""),
        "category": req.get("category", ""),
        "description": req.get("description", ""),
    }
    if entry:
        out["entry"] = entry
    out["cases"] = cases
    return out


def simple_to_zephyr(simple: dict, pdf_name: str = "", page: int | str = "") -> dict:
    """단순 포맷 → Zephyr req (결정적 조립).

    각 case → 1 step: data=[Step]\\n1.{check}, result=[DUX]\\n번호목록(expect).
    네비게이션 prefix는 일정 템플릿 없이 생략(추후 화면 진입 prec로 보강 가능).
    """
    ver = f"*[Ver.]*\n{pdf_name} - {page}p" if pdf_name else "*[Ver.]*"
    entry = simple.get("entry", []) or []
    # case 단위 반복 억제: (cond,check,expect) 정규화 중복 제거 (모델이 같은 case 반복하는 증상)
    seen_cases: set = set()
    steps = []
    for c in simple.get("cases", []):
        check = c.get("check", "").strip()
        expect = c.get("expect", "").strip()
        cond = c.get("cond", "").strip()
        ckey = (cond, " ".join(check.split()), " ".join(expect.split()))
        if ckey in seen_cases:
            continue
        seen_cases.add(ckey)
        # data = 네비게이션 entry + 확인동작 check (번호 매김)
        nav = [x.strip() for x in entry if x.strip()]
        data_lines = nav + ([check] if check else [])
        data = "*[Step]*\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(data_lines, 1))
        # result = expect 원형 보존 (이미 번호/계층 포함)
        if expect and re.match(r"^\d", expect):
            result = "*[DUX]*\n" + expect
        else:
            ex_lines = [x.strip() for x in expect.split("\n") if x.strip()]
            result = "*[DUX]*\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(ex_lines, 1))
        step_field = ver
        if cond:
            cond_lines = [x.strip() for x in cond.split("/") if x.strip()]
            step_field += "\n\n*[Precondition]*\n" + "\n".join(
                f"{i}. {x}" for i, x in enumerate(cond_lines, 1))
        steps.append({
            "step": step_field,
            "data": data,
            "result": result,
        })
    return {
        "id": simple.get("id", ""),
        "category": simple.get("category", ""),
        "description": simple.get("description", ""),
        "testable": True,
        "priority": "medium",
        "page": str(page) if page != "" else "",
        "zephyr_tc": {"summary": simple.get("description", ""), "steps": steps},
    }
