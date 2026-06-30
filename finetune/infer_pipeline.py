#!/usr/bin/env python3
"""
기획서 PDF → Zephyr TC 생성 파이프라인

Step 1: Qwen2.5-VL-7B  — PDF 페이지 이미지 → 텍스트 OCR
Step 2: EXAONE 3.5 7.8B + LoRA — OCR 텍스트 → Zephyr TC JSON

사용법:
  # PDF 파일로 TC 생성
  python3 infer_pipeline.py --pdf /path/to/spec.pdf --pages 3,5,7

  # 텍스트 파일로 TC 생성 (OCR 없이)
  python3 infer_pipeline.py --spec spec.txt

  # OCR 결과만 확인
  python3 infer_pipeline.py --pdf /path/to/spec.pdf --pages 3 --ocr-only
"""

import argparse
import json
import pathlib
import re
import sys
from typing import Optional

# ── 모델 설정 ──────────────────────────────────────────────────────────────────
# HF repo ID 사용 — 로컬 캐시 자동 탐색, 없으면 자동 다운로드 (머신 간 휴대 가능)
VLM_PATH = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
EXAONE_PATH = "mlx-community/EXAONE-3.5-7.8B-Instruct-4bit"
ADAPTER_PATH = pathlib.Path(__file__).parent / "adapters_best"

# ── OCR 프롬프트 ───────────────────────────────────────────────────────────────
# 절충형 OCR 프롬프트: prepare_data.py의 VLM_OCR_PROMPT와 반드시 동일하게 유지 (학습=추론 일치)
# OCR강화(v20) — prepare_data.py VLM_OCR_PROMPT와 반드시 동일 (train=infer)
OCR_PROMPT = (
    "이 기획서 페이지의 텍스트를 빠짐없이 그대로 옮겨줘.\n"
    "특히 화면 우측의 상세 설명(Description/스펙) 표가 핵심이다 — 한 항목도 빼지 말고 전부 포함해.\n"
    "각 항목의 번호·계층(1, 1-1, 2-1 등)과 구체적 값(색상, 위치, 개수, 문구, 조건, 동작, 결과)을 그대로 적어.\n"
    "화면 목업 안의 UI 라벨·버튼명·상태 텍스트도 포함해.\n"
    "요약·생략 금지. 단, 완전히 동일한 문구가 여러 번 반복되면 1개만 남겨.\n"
    "마크다운 기호(#, *)나 장식 없이 번호 목록으로 작성해."
)


def dedup_ocr_text(text: str, max_repeat: int = 2) -> str:
    """OCR 출력 후처리(B): 마크다운 제거 + 반복 라인 합치기.
    prepare_data.py의 dedup_ocr_text와 동일 로직 (학습=추론 일치)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)
        line = re.sub(r"^[\s>#*\-]+", "", line)
        line = re.sub(r"(.{4,40}?)(?:\1){2,}", r"\1", line)  # 줄내 구절 반복 붕괴 합침
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\[p\.\d+\]$", line):
            out.append(line)
            continue
        key = re.sub(r"^[\d.\-]+\s*", "", re.sub(r"\s+", " ", line).lower())  # 번호접두사 제거 후 dedup
        cnt = seen.get(key, 0)
        seen[key] = cnt + 1
        if cnt >= max_repeat:
            continue
        out.append(line)
    return "\n".join(out)

# ── EXAONE 시스템 프롬프트 ────────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 모바일 앱 테스트케이스 작성 전문가입니다.
기획서 텍스트에서 테스트 가능한 요구사항을 빠짐없이 추출하여 JSON 형식으로 반환합니다.

【필수 추출 원칙】
1. 기획서에 정의된 모든 항목을 누락 없이 포함한다.
   - 화면별 UI 요소, 상태, 조건, 표시 텍스트/포맷 규칙을 전부 TC로 작성한다.
   - "N개", "말줄임", "이미지 없을 때" 등 구체적 케이스가 언급된 경우 각각 별도 TC로 만든다.
2. 기획서 내용으로 예상 가능한 예외·경계 케이스도 포함한다.
   - 빈 목록 / 데이터 없음 상태
   - 최대·최소 값, 긴 텍스트 말줄임
   - 네트워크 오류 / 로딩 실패
   - 권한 없음 / 로그인 필요 상태
   - 중복 입력, 잘못된 입력 형식

추출 항목:
- UI 요소와의 상호작용 (탭, 스와이프, 입력 등)
- 화면 전환 및 네비게이션
- 데이터 표시 규칙 (포맷, 말줄임, 이미지 유무 등)
- 상태별 동작 (로딩 중, 완료, 오류 등)
- 예외·경계 케이스

반드시 유효한 JSON만 응답하세요. 모든 텍스트 값은 반드시 한글로 작성하세요."""


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: PDF 텍스트 추출 (학습 데이터와 동일한 pymupdf 방식)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_pages(pdf_path: str, pages: list[int]) -> str:
    """pymupdf로 PDF 텍스트 추출 (학습 데이터와 동일한 형식)."""
    try:
        import fitz
    except ImportError:
        print("❌ pymupdf 없음: pip install pymupdf")
        sys.exit(1)

    print(f"\n📄 텍스트 추출: {pdf_path} (페이지: {pages})")
    doc = fitz.open(pdf_path)
    total = len(doc)
    parts = []

    for page_num in pages:
        print(f"  📖 p.{page_num} 추출 중...", end=" ", flush=True)
        if not (1 <= page_num <= total):
            print(f"⚠️  범위 초과 (총 {total}p)")
            continue
        text = doc[page_num - 1].get_text().strip()
        if len(text) < 100:
            print(f"⚠️  텍스트 부족({len(text)}자) — 이미지 PDF일 수 있음")
        else:
            print(f"✅ ({len(text)}자)")
        parts.append(f"[p.{page_num}]\n{text}")

    doc.close()
    return "\n\n".join(parts)


def _collapse_repeated_phrases(text: str, min_len: int = 5, max_len: int = 40,
                               keep: int = 2) -> str:
    """목업 더미텍스트 방어: 같은 짧은 구절이 공백/줄바꿈 사이로 keep회 초과 연속 반복되면
    keep회만 남김. 예: '추론 모델 분석 내용'×수십, '질문에대한답변'×100 → 2회.
    정상 콘텐츠(반복 없음)는 무손상. 모델이 더미를 echo→파싱붕괴하는 페이지 안정화."""
    pat = re.compile(r'(.{%d,%d}?)(?:\s*\1){%d,}' % (min_len, max_len, keep), re.DOTALL)
    prev = None
    while prev != text:
        prev = text
        text = pat.sub(lambda m: (m.group(1) + ' ') * keep, text)
    return text


def _looks_clean(text: str) -> bool:
    """pymupdf 텍스트가 정상 인코딩인지(모지베이크 아닌지) 판정.

    일부 PDF는 서브셋/커스텀 폰트라 pymupdf가 깨진 글자(Ÿ°ð˝ 등)를 뱉음 → 이 경우
    VLM OCR이 나음. 한글·영숫자·기본문장부호 비율로 구분(정상≈0.99 vs 깨짐≈0.65~0.73).
    """
    if not text:
        return False
    good = len(re.findall(r'[가-힣A-Za-z0-9\s.,:/()\[\]·\-→%]', text))
    return good / len(text) >= 0.85


def ocr_pdf_pages(pdf_path: str, pages: list[int], max_size: int = 1200,
                  text_threshold: int = 150) -> str:
    """하이브리드 추출: 텍스트 PDF는 pymupdf(완전·정확), 이미지 PDF 페이지만 VLM OCR.

    배경: VLM OCR(1.5x)이 텍스트 PDF의 표 행을 누락하는 손실 확인(Side Drawer p.10 '설정'
    행 통째 누락 → 케이스 부실). pymupdf는 임베디드 텍스트를 직접 읽어 완전. 텍스트가 더
    완전하므로 grounding에도 유리. 스캔/이미지 페이지(임베디드 텍스트<threshold)만 VLM 폴백.
    """
    import fitz
    import io

    print(f"\n📄 텍스트 추출(하이브리드): {pdf_path} (페이지: {pages})")
    doc = fitz.open(pdf_path)
    total = len(doc)
    results: dict[int, str] = {}
    need_vlm: list[int] = []

    # 1) pymupdf 우선 — 임베디드 텍스트 충분한 페이지는 그대로 사용(완전·정확)
    for page_num in pages:
        if not (1 <= page_num <= total):
            print(f"  ⚠️  p.{page_num} 범위 초과(총 {total}p)")
            continue
        txt = doc[page_num - 1].get_text().strip()
        if len(txt) >= text_threshold and _looks_clean(txt):
            print(f"  📖 p.{page_num} pymupdf ✅ ({len(txt)}자)")
            results[page_num] = txt
        else:
            why = "텍스트 부족" if len(txt) < text_threshold else "폰트 깨짐(모지베이크)"
            print(f"  📖 p.{page_num} {why}({len(txt)}자) → VLM OCR 폴백")
            need_vlm.append(page_num)

    # 2) 텍스트 부족(스캔/이미지) 페이지만 VLM OCR
    if need_vlm:
        from mlx_vlm import load
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        print(f"🔄 Qwen2.5-VL-7B 로드 중... (이미지 PDF {len(need_vlm)}p)")
        model, processor = load(str(VLM_PATH))
        config = model.config.__dict__
        prompt_text = apply_chat_template(processor, config, OCR_PROMPT, num_images=1)

        for page_num in need_vlm:
            print(f"  📖 p.{page_num} VLM OCR 중...", end=" ", flush=True)
            pix = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = ""
            for chunk in stream_generate(model, processor, prompt_text, [img], max_tokens=2500):
                text += chunk.text if hasattr(chunk, "text") else str(chunk)
            cleaned = dedup_ocr_text(text.strip())
            print(f"✅ ({len(cleaned)}자)")
            results[page_num] = cleaned

    doc.close()
    # 요청 페이지 순서대로 [p.N] 마커 붙여 조립 (목업 더미 반복구절 접기)
    return "\n\n".join(
        f"[p.{p}]\n{_collapse_repeated_phrases(results[p])}"
        for p in pages if p in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: TC 생성 — 텍스트 → Zephyr TC JSON
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tc(spec_text: str, adapter_path: str = str(ADAPTER_PATH),
                max_tokens: int = 2000, repetition_penalty: float = 1.0,
                temp: float = 0.5, simple_mode: bool = True,
                pdf_name: str = "", page: str = "", best_of: int = 1) -> dict:
    """EXAONE LoRA로 TC 생성.

    simple_mode(Phase4, 기본): 모델은 단순포맷({cases:[{cond,check,expect}]})만 생성,
      simple_to_zephyr로 결정적 조립 → 복잡 JSON 직접생성 시의 폭주 회피.
    """
    import gc
    from mlx_lm import load, generate

    print(f"\n🤖 TC 생성 시작")
    print(f"   모델: EXAONE 3.5 7.8B + LoRA ({adapter_path})")
    print(f"   입력 텍스트 앞 100자: {spec_text[:100]!r}")

    # 이전 세션의 Metal KV 캐시 잔재를 제거
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

    model, tokenizer = load(
        str(EXAONE_PATH),
        adapter_path=adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )

    if simple_mode:
        from simple_format import SIMPLE_SYSTEM_PROMPT
        sys_prompt = SIMPLE_SYSTEM_PROMPT
    else:
        sys_prompt = SYSTEM_PROMPT

    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    logits_processors = make_logits_processors(
        repetition_penalty=repetition_penalty, repetition_context_size=40
    ) if repetition_penalty and repetition_penalty != 1.0 else None

    # 응답에서 JSON 객체 추출 후 파싱 (아래 헬퍼는 모델 비의존 — 페이지마다 재사용)
    def _derepeat(text: str, min_len: int = 10, max_repeat: int = 3) -> str:
        """JSON 문자열 값 안에서 반복 패턴을 검출해 첫 출현만 남김."""
        lines = text.split("\\n")
        seen: dict[str, int] = {}
        result = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) >= min_len:
                cnt = seen.get(stripped, 0)
                seen[stripped] = cnt + 1
                if cnt >= max_repeat:
                    continue
            result.append(line)
        return "\\n".join(result)

    def _find_safe_truncation(text: str) -> str:
        """JSON string 안을 추적해 마지막으로 안전하게 자를 수 있는 위치를 반환.

        우선순위:
        1. 마지막으로 depth≥1 인 }] 닫힌 위치
        2. 없으면 마지막으로 문자열 바깥인 위치 (key-value 사이)
        """
        depth = 0
        in_string = False
        escape = False
        last_bracket_close = 0   # depth≥1인 }] 닫힌 위치
        last_outside_string = 0  # 문자열 바깥 마지막 위치
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                if not in_string:
                    last_outside_string = i + 1
                continue
            if not in_string:
                last_outside_string = i + 1
                if ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth >= 1:
                        last_bracket_close = i + 1
        if last_bracket_close:
            return text[:last_bracket_close]
        if last_outside_string:
            return text[:last_outside_string]
        return text

    def _strip_line_comments(t: str) -> str:
        """문자열 밖의 // 줄주석 제거 (모델 환각 대비). URL(https://)은 문자열 안이라 보존."""
        res = []
        in_str = False
        esc = False
        i = 0
        n = len(t)
        while i < n:
            ch = t[i]
            if in_str:
                res.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
                res.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < n and t[i + 1] == "/":
                # 줄 끝까지 건너뜀
                while i < n and t[i] != "\n":
                    i += 1
                continue
            res.append(ch)
            i += 1
        return "".join(res)

    def _repair_and_parse(raw: str):
        """JSON 파싱 시도. 실패 시 //주석 제거·잘린 문자열·괄호 처리 후 재시도."""
        raw = _strip_line_comments(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 괄호 카운팅으로 닫기
        trunc = raw.rstrip(",\n ")
        ob  = trunc.count("{") - trunc.count("}")
        ob2 = trunc.count("[") - trunc.count("]")
        closing = "]" * max(ob2, 0) + "}" * max(ob, 0)
        if closing:
            try:
                return json.loads(trunc + closing)
            except json.JSONDecodeError:
                pass

        # 잘린 JSON 문자열 처리: 마지막 안전 위치까지 자르고 재시도
        safe = _find_safe_truncation(raw)
        if safe and len(safe) < len(raw):
            s = safe.rstrip(",\n ")
            ob  = s.count("{") - s.count("}")
            ob2 = s.count("[") - s.count("]")
            closing = "]" * max(ob2, 0) + "}" * max(ob, 0)
            try:
                return json.loads(s + closing)
            except json.JSONDecodeError:
                pass
        return None

    def _richness(d: dict) -> int:
        """후보 우선순위 — simple_mode(cases/screens)와 full(requirements) 모두 대응."""
        if not isinstance(d, dict):
            return -1
        if isinstance(d.get("screens"), list):
            return sum(len(s.get("cases", [])) for s in d["screens"])
        return len(d.get("cases", [])) + len(d.get("requirements", []))

    def _balanced_objects(t: str):
        """머리말/펜스 무시하고 텍스트의 모든 최상위 {...} 후보를 추출 (잘린 것 포함)."""
        objs = []
        i = 0
        n = len(t)
        while i < n:
            if t[i] != "{":
                i += 1
                continue
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                ch = t[j]
                if esc:
                    esc = False
                elif ch == "\\" and in_str:
                    esc = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            objs.append(t[i:j + 1])   # 닫혔으면 완전, 안 닫혔으면 끝까지(잘림→repair)
            i = j + 1
        return objs

    def _collapse_runaway(text: str) -> str:
        """JSON 값 안의 반복 폭주(예: '질문에대한답변'×100, '추론 모델 분석 내용'×수십)를
        붕괴 — 폰트깨짐/모지베이크 OCR을 모델이 echo해 파싱이 깨지는 케이스 방어.
        4~60자 단위가 3회 이상 연속 반복(사이 공백 허용)되면 2회만 남김."""
        prev = None
        cur = text
        # 공백 포함 반복(추론 모델 분석 내용 …)과 무공백 반복(질문에대한답변…) 모두 대상
        for _ in range(3):  # 중첩 반복 대비 수렴까지 몇 회
            if cur == prev:
                break
            prev = cur
            # 4회 이상 연속 반복만 붕괴(정상 3회 패턴은 보존), 2회만 남김
            cur = re.sub(r"(.{4,60}?)(?:\s*\1){3,}", r"\1\1", cur)
        return cur

    def try_parse(text: str):
        # 머리말("다음은 …") + ```json/``` 펜스 제거 + 반복 폭주 붕괴
        stripped = re.sub(r"```(?:json)?", "", text)
        stripped = _collapse_runaway(stripped)
        for src in (_derepeat(stripped), stripped, text):
            candidates = []
            for block in _balanced_objects(src):
                parsed = _repair_and_parse(block.strip())
                if isinstance(parsed, dict) and (_richness(parsed) > 0 or "description" in parsed):
                    candidates.append(parsed)
            if candidates:
                return max(candidates, key=_richness)
        return None

    # ── 단일 페이지(chunk) 생성 + 파싱 ──
    def _gen_one(chunk_text: str):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": (
                f"다음 기획서에서 테스트케이스를 작성해주세요.\n\n"
                f"주의사항:\n"
                f"- 기획서에 정의된 모든 항목을 누락 없이 포함하세요.\n"
                f"- 기획서 내용으로 예상 가능한 예외 케이스(빈 상태, 긴 텍스트, 오류 상황 등)도 포함하세요.\n\n"
                f"기획서:\n{chunk_text}"
            )},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        def _gen(t: float) -> str:
            print(f"   생성 중... (temp={t}, rep_penalty={repetition_penalty})")
            return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                            verbose=False, sampler=make_sampler(temp=t, top_p=0.9),
                            logits_processors=logits_processors)

        # 파싱 실패 시 greedy(temp=0)로 재시도 — 깨진 JSON은 보통 높은 temp 샘플링 운빨
        response = _gen(temp)
        result = try_parse(response)
        if result is None and temp > 0.0:
            print("   ⚠️  파싱 실패 — greedy(temp=0)로 재시도")
            response = _gen(0.0)
            result = try_parse(response)
        return result, response

    # ── 페이지 단위 분할 생성 ──
    # ocr_pdf_pages가 각 페이지 앞에 [p.N] 마커를 붙임. 마커가 2개 이상이면
    # 페이지별로 따로 생성해 병합한다(모델이 1페이지 단위로 학습돼, 여러 페이지를
    # 한 번에 넣으면 앞부분만 처리되는 문제 회피).
    page_matches = list(re.finditer(r"\[p\.(\d+)\]\s*", spec_text))
    chunks = []  # (페이지라벨, 본문)
    if len(page_matches) >= 2:
        for idx, m in enumerate(page_matches):
            start = m.end()
            end = page_matches[idx + 1].start() if idx + 1 < len(page_matches) else len(spec_text)
            body = spec_text[start:end].strip()
            if body:
                chunks.append((m.group(1), body))
    if not chunks:
        chunks = [(page, spec_text)]

    if len(chunks) > 1:
        print(f"   📑 페이지 단위 생성: {len(chunks)}개 페이지")

    merged_reqs = []
    last_raw = ""
    diag = {}  # 페이지별 {모델 원시 case 수, 최종(dedup·조립 후) step 수} — 부실 원인 진단
    for pg_label, chunk_text in chunks:
        if len(chunks) > 1:
            print(f"   ── p.{pg_label} TC 생성 ──")
        if best_of > 1 and simple_mode:
            # best-of-N: N회 생성 → case 합집합(dedup)으로 커버리지↑
            gens = [_gen_one(chunk_text) for _ in range(best_of)]
            valids = [r for r, _ in gens if r]
            result = _union_simple(valids) if valids else None
            response = gens[-1][1] if gens else ""
        else:
            result, response = _gen_one(chunk_text)
        last_raw = response
        if not result:
            continue
        if simple_mode:
            raw_n = _simple_case_count(result)
            assembled = _assemble_simple(result, pdf_name, str(pg_label))
            reqs = assembled.get("requirements", [])
            final_n = sum(len(r.get("zephyr_tc", {}).get("steps", [])) for r in reqs)
            diag[str(pg_label)] = {"raw_cases": raw_n, "final_cases": final_n}
            print(f"      모델 원시 case {raw_n}개 → 최종 {final_n}개")
            merged_reqs.extend(reqs)
        else:
            merged_reqs.extend(result.get("requirements", [result]))

    # 모델 해제 (다음 호출에 KV 캐시 잔재 방지)
    del model, tokenizer
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

    if merged_reqs:
        print(f"   ✅ JSON 파싱 성공 — 총 {len(merged_reqs)}개 TC ({len(chunks)}개 페이지)")
        return {"requirements": merged_reqs,
                "total_count": len(merged_reqs),
                "testable_count": len(merged_reqs),
                "_diag": diag}

    print("   ⚠️  JSON 파싱 실패 — raw 텍스트 반환")
    return {"raw": last_raw}


def _union_simple(results: list):
    """best-of-N: 여러 생성 결과의 case를 dedup union → 단일 simple dict.

    같은 페이지를 N회 생성하면 매번 다른 case가 나옴 → 합집합으로 커버리지↑.
    dedup 키 = (cond, 정규화 check, 정규화 expect)로 simple_to_zephyr와 동일.
    """
    seen, cases, base = set(), [], None
    for r in results:
        if not isinstance(r, dict):
            continue
        screens = r.get("screens") if isinstance(r.get("screens"), list) else [r]
        for s in screens:
            if not isinstance(s, dict):
                continue
            if base is None:
                base = {k: s.get(k, "") for k in ("id", "category", "description", "entry")}
            for c in s.get("cases", []) or []:
                check = (c.get("check", "") or "").strip()
                expect = (c.get("expect", "") or "").strip()
                cond = (c.get("cond", "") or "").strip()
                if not (check or expect):
                    continue
                key = (cond, " ".join(check.split()), " ".join(expect.split()))
                if key in seen:
                    continue
                seen.add(key)
                cases.append(c)
    if base is None:
        return None
    base["cases"] = cases
    return base


def _simple_case_count(parsed: dict) -> int:
    """모델 원시 출력의 case 총수 (dedup/조립 전). screens 배열도 합산."""
    if isinstance(parsed.get("screens"), list):
        return sum(len(s.get("cases", []) or []) for s in parsed["screens"])
    return len(parsed.get("cases", []) or [])


def _assemble_simple(parsed: dict, pdf_name: str = "", page: str = "") -> dict:
    """단순포맷 모델 출력({cases} 또는 {screens:[...]}) → Zephyr {requirements:[...]}."""
    from simple_format import simple_to_zephyr
    simples = parsed.get("screens") if isinstance(parsed.get("screens"), list) else [parsed]
    reqs = [simple_to_zephyr(s, pdf_name, page) for s in simples if s.get("cases")]
    return {"requirements": reqs, "total_count": len(reqs),
            "testable_count": len(reqs)}


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="기획서 PDF → Zephyr TC 생성 파이프라인")

    # 입력
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",  type=str, help="기획서 PDF 경로")
    group.add_argument("--spec", type=str, help="기획서 텍스트 파일 경로 (OCR 없이)")

    parser.add_argument("--pages", type=str, default="3",
                        help="PDF 페이지 번호 (콤마 구분, 예: 3,5,7)")
    parser.add_argument("--vlm-ocr", action="store_true", default=True,
                        help="VLM OCR 사용 (기본: True, v5 학습 데이터 형식과 일치)")
    parser.add_argument("--no-vlm-ocr", dest="vlm_ocr", action="store_false",
                        help="pymupdf 텍스트 추출 사용 (VLM OCR 비활성화)")
    parser.add_argument("--ocr-only", action="store_true",
                        help="텍스트 추출 결과만 출력 (TC 생성 건너뜀)")
    parser.add_argument("--adapter", type=str, default=str(ADAPTER_PATH),
                        help="EXAONE LoRA 어댑터 경로")
    parser.add_argument("--output", type=str, help="결과 저장 JSON 파일 경로")
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="반복 루프 억제 (1.0=비활성, 권장 1.15~1.3)")

    args = parser.parse_args()

    # ── Step 1: 기획서 텍스트 획득 ────────────────────────────────────────────
    if args.pdf:
        pages = [int(p.strip()) for p in args.pages.split(",")]
        if args.vlm_ocr:
            spec_text = ocr_pdf_pages(args.pdf, pages)
        else:
            spec_text = extract_pdf_pages(args.pdf, pages)
    else:
        with open(args.spec, encoding="utf-8") as f:
            spec_text = f.read()
        print(f"📄 기획서 파일 로드: {args.spec} ({len(spec_text)}자)")

    print("\n" + "=" * 60)
    print("📝 OCR 결과 (앞 500자):")
    print(spec_text[:500])
    print("=" * 60)

    if args.ocr_only:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(spec_text)
            print(f"\n💾 OCR 결과 저장: {args.output}")
        return

    # ── Step 2: TC 생성 ────────────────────────────────────────────────────────
    result = generate_tc(spec_text, args.adapter, args.max_tokens,
                         repetition_penalty=args.repetition_penalty)

    print("\n" + "=" * 60)
    print("📋 생성된 TC:")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:3000])
    print("=" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
