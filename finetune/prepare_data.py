#!/usr/bin/env python3
"""
파인튜닝 학습 데이터 준비 스크립트

입력: 기획서(spec) + 테스트케이스(TC) 쌍 JSON 파일들
출력: MLX-LM LoRA 학습용 train.jsonl / valid.jsonl

사용법:
  python prepare_data.py --input-dir ./raw_data --output-dir ./data

raw_data 디렉토리 구조:
  raw_data/
    pair_001.json
    pair_002.json
    ...

각 JSON 파일 형식:
  {
    "spec": "기획서 원문 텍스트 (관련 섹션)",
    "requirements": [
      {
        "id": "REQ-001",
        "category": "기능",
        "description": "요구사항 설명",
        "priority": "high",
        "acceptance_criteria": ["기준1", "기준2"]
      }
    ]
  }

기존 tc_creation_examples/ 형식도 자동 변환 지원:
  {
    "requirement": "요구사항 텍스트",
    "tc": { "title": "...", "steps": [...] }
  }
"""

import json
import os
import re
import random
import argparse
from pathlib import Path

# 시스템 프롬프트 (파인튜닝 후 추론 시에도 동일하게 사용 — infer_pipeline.py와 반드시 동기화)
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

# Claude Vision API key (이미지 PDF 처리용 — 없으면 텍스트 추출만 시도)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VISION_MODEL = "claude-sonnet-4-6"


# ─── PDF → spec 텍스트 추출 ────────────────────────────────────────────────────

def _pdf_text_extract(pdf_path: str, pages: list[int] | None) -> str:
    """pymupdf로 PDF 텍스트 추출 (무료, API 불필요)"""
    import fitz
    doc = fitz.open(pdf_path)
    total = len(doc)
    targets = [p - 1 for p in pages if 1 <= p <= total] if pages else range(total)
    parts = []
    for i in targets:
        text = doc[i].get_text().strip()
        if text:
            parts.append(f"[p.{i + 1}]\n{text}")
    doc.close()
    return "\n\n".join(parts)


def _pdf_vision_extract(pdf_path: str, pages: list[int] | None, api_key: str) -> str:
    """Claude Vision으로 이미지 PDF 추출 (API key 필요)"""
    import base64
    import fitz
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    doc = fitz.open(pdf_path)
    total = len(doc)
    targets = [p - 1 for p in pages if 1 <= p <= total] if pages else range(total)

    prompt = ("이 기획서 페이지의 내용을 테스트 케이스 작성에 필요한 형태로 정리해주세요. "
              "UI 요소, 동작 조건, 기능 설명, 사용자 흐름을 원문 그대로 포함해주세요. 요약하지 마세요.")

    parts = []
    for i in targets:
        mat = fitz.Matrix(1.5, 1.5)
        pix = doc[i].get_pixmap(matrix=mat)
        img_b64 = base64.standard_b64encode(pix.tobytes("png")).decode()
        msg = client.messages.create(
            model=VISION_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png",
                                             "data": img_b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        parts.append(f"[p.{i + 1}]\n{msg.content[0].text.strip()}")

    doc.close()
    return "\n\n".join(parts)


_VLM_MODEL = None
_VLM_PROCESSOR = None

# 페이지 단위 OCR 결과 캐시 — (abs_pdf_path, page_num_str) → ocr_text
# 파일로 저장하여 중단 후 재시작 시에도 재사용
_OCR_CACHE: dict[str, str] = {}
_OCR_CACHE_FILE: Path | None = None


def _ocr_cache_key(pdf_path: str, page: int | str) -> str:
    # 파일명 기준 (머신 간 경로 달라도 캐시 재사용 — 회사/집 동일 PDF 공유)
    return f"{Path(pdf_path).name}::{page}"


def load_ocr_cache(output_dir: Path):
    """output_dir/ocr_cache.json 에서 캐시 로드."""
    global _OCR_CACHE, _OCR_CACHE_FILE
    _OCR_CACHE_FILE = output_dir / "ocr_cache.json"
    if _OCR_CACHE_FILE.exists():
        _OCR_CACHE = json.loads(_OCR_CACHE_FILE.read_text(encoding="utf-8"))
        print(f"  💾 OCR 캐시 로드: {len(_OCR_CACHE)}개 항목")
    else:
        _OCR_CACHE = {}


def save_ocr_cache():
    """현재 캐시를 파일에 저장."""
    if _OCR_CACHE_FILE:
        _OCR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _OCR_CACHE_FILE.write_text(
            json.dumps(_OCR_CACHE, ensure_ascii=False, indent=2), encoding="utf-8"
        )

# 절충형 OCR 프롬프트: 내용 완전성(verbatim 장점) + 적당한 길이(반복 제거)
# 학습/추론 동일 프롬프트 사용 — infer_pipeline.py의 OCR_PROMPT와 반드시 일치시킬 것
# OCR강화(v20): grounding↑ — 우측 상세 Description/스펙 표를 빠짐없이 캡처.
# infer_pipeline.py의 OCR_PROMPT와 반드시 동일 (train=infer).
VLM_OCR_PROMPT = (
    "이 기획서 페이지의 텍스트를 빠짐없이 그대로 옮겨줘.\n"
    "특히 화면 우측의 상세 설명(Description/스펙) 표가 핵심이다 — 한 항목도 빼지 말고 전부 포함해.\n"
    "각 항목의 번호·계층(1, 1-1, 2-1 등)과 구체적 값(색상, 위치, 개수, 문구, 조건, 동작, 결과)을 그대로 적어.\n"
    "화면 목업 안의 UI 라벨·버튼명·상태 텍스트도 포함해.\n"
    "요약·생략 금지. 단, 완전히 동일한 문구가 여러 번 반복되면 1개만 남겨.\n"
    "마크다운 기호(#, *)나 장식 없이 번호 목록으로 작성해."
)


def dedup_ocr_text(text: str, max_repeat: int = 2) -> str:
    """OCR 출력 후처리 (B): 마크다운 장식 제거 + 반복 라인 합치기.

    - **굵게**, 머리 기호(#, *, -, >), 과도한 들여쓰기 제거
    - 정규화된 동일 라인이 max_repeat회를 넘으면 이후 중복 제거
      (verbatim 시 발생하던 "...생성 중" x50 반복 루프 방지)
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)   # **bold** → bold
        line = re.sub(r"^[\s>#*\-]+", "", line)          # 머리 기호/들여쓰기 제거
        # 한 줄 안에서 같은 구절(4~40자)이 3회+ 연속 반복되면 1회로 (VLM 줄내 폭주)
        line = re.sub(r"(.{4,40}?)(?:\1){2,}", r"\1", line)
        line = line.strip()
        if not line:
            continue
        # [p.N] 페이지 헤더는 항상 유지
        if re.match(r"^\[p\.\d+\]$", line):
            out.append(line)
            continue
        # dedup 키: 앞 번호접두사(1.1.2.1. / 1-2 / 1) 제거 → 번호만 증가하는 반복도 합침
        key = re.sub(r"^[\d.\-]+\s*", "", re.sub(r"\s+", " ", line).lower())
        cnt = seen.get(key, 0)
        seen[key] = cnt + 1
        if cnt >= max_repeat:
            continue
        out.append(line)
    return "\n".join(out)


def _load_vlm_once():
    """Qwen2.5-VL-7B를 한 번만 로드 (전역 캐시)."""
    global _VLM_MODEL, _VLM_PROCESSOR
    if _VLM_MODEL is None:
        from mlx_vlm import load as vlm_load
        VLM_HF_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
        VLM_PATH = (Path.home() / ".cache/huggingface/hub"
                    / "models--mlx-community--Qwen2.5-VL-7B-Instruct-4bit")
        snapshots = list((VLM_PATH / "snapshots").glob("*")) if (VLM_PATH / "snapshots").exists() else []
        model_path = str(snapshots[0]) if snapshots else VLM_HF_ID
        print(f"\n🔄 Qwen2.5-VL-7B 로드 중... ({model_path})")
        _VLM_MODEL, _VLM_PROCESSOR = vlm_load(model_path)
        print("✅ VLM 로드 완료")
    return _VLM_MODEL, _VLM_PROCESSOR


def _pdf_vlm_ocr_extract(pdf_path: str, pages: list[int] | None) -> str:
    """Qwen2.5-VL-7B로 PDF 페이지 OCR (깨끗한 한글 추출). 페이지 단위 캐시 활용."""
    import fitz
    from PIL import Image
    import io

    abs_path = str(Path(pdf_path).resolve())
    doc = fitz.open(pdf_path)
    total = len(doc)
    targets = [p - 1 for p in pages if 1 <= p <= total] if pages else range(total)
    parts = []
    needs_vlm = []

    # 캐시 히트 먼저 처리
    for i in targets:
        key = _ocr_cache_key(abs_path, i + 1)
        if key in _OCR_CACHE:
            parts.append((i, _OCR_CACHE[key]))
        else:
            needs_vlm.append(i)

    if needs_vlm:
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor = _load_vlm_once()
        config = model.config.__dict__
        prompt_text = apply_chat_template(processor, config, VLM_OCR_PROMPT, num_images=1)

        try:
            import mlx.core as _mx
        except Exception:
            _mx = None

        for i in needs_vlm:
            mat = fitz.Matrix(1.5, 1.5)
            pix = doc[i].get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            text = ""
            for chunk in stream_generate(model, processor, prompt_text, [img], max_tokens=2500):
                t = chunk.text if hasattr(chunk, "text") else str(chunk)
                text += t

            # 후처리(B): 마크다운/반복 제거
            cleaned = dedup_ocr_text(text.strip())
            page_text = f"[p.{i + 1}]\n{cleaned}"
            key = _ocr_cache_key(abs_path, i + 1)
            _OCR_CACHE[key] = page_text
            save_ocr_cache()
            parts.append((i, page_text))

            # M1 16GB 연속 OCR OOM 방지: 페이지마다 Metal 캐시 해제
            if _mx is not None:
                _mx.clear_cache()

    doc.close()
    parts.sort(key=lambda x: x[0])
    return "\n\n".join(text for _, text in parts)


def load_spec_from_pdf(pdf_path: str, pages: list[int] | None = None,
                       api_key: str = "", use_vlm_ocr: bool = False) -> str:
    """
    PDF 파일에서 spec 텍스트 추출

    우선순위:
      1. pymupdf 텍스트 추출 (무료)
      2. 텍스트가 너무 적으면 Claude Vision (api_key 필요)

    Args:
        pdf_path : PDF 파일 경로 (절대 또는 상대)
        pages    : 1-base 페이지 번호 목록 (None = 전체)
        api_key  : Anthropic API key (이미지 PDF 처리용)
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        print(f"    ❌ PDF 없음: {path}")
        return ""

    page_label = f"p.{pages}" if pages else "전체"
    print(f"    📄 {path.name} ({page_label}) 텍스트 추출 중...", end="", flush=True)

    # ① VLM OCR (--vlm-ocr 옵션)
    if use_vlm_ocr:
        try:
            print(" VLM OCR...", end="", flush=True)
            text = _pdf_vlm_ocr_extract(str(path), pages)
            print(f" ✅ ({len(text)}자, VLM OCR)")
            return text
        except Exception as e:
            print(f" ⚠️  VLM OCR 실패: {e}, pymupdf로 대체")

    # ② pymupdf 텍스트 시도
    try:
        text = _pdf_text_extract(str(path), pages)
        if len(text) >= 200:
            print(f" ✅ ({len(text)}자, 텍스트 추출)")
            return text
        # 텍스트가 너무 적으면 이미지 PDF로 간주
        sparse_text = text
    except ImportError:
        print()
        print("    ⚠️  pymupdf 없음 → pip install pymupdf")
        return ""
    except Exception as e:
        print(f" ⚠️  텍스트 추출 실패: {e}")
        sparse_text = ""

    # ② Claude Vision 시도 (이미지 PDF)
    _key = api_key or ANTHROPIC_API_KEY
    if _key:
        try:
            print(f" 텍스트 부족 → Vision API 사용...", end="", flush=True)
            text = _pdf_vision_extract(str(path), pages, _key)
            print(f" ✅ ({len(text)}자, Vision)")
            return text
        except ImportError:
            print("\n    ⚠️  anthropic 없음 → pip install anthropic")
        except Exception as e:
            print(f" ❌ Vision 오류: {e}")
    else:
        if sparse_text:
            print(f" ⚠️  텍스트 부족({len(sparse_text)}자). 이미지 PDF라면 ANTHROPIC_API_KEY 설정 권장")
            return sparse_text
        print(f"\n    ⚠️  이미지 PDF입니다. ANTHROPIC_API_KEY를 설정하거나 spec을 직접 입력하세요.")

    return sparse_text


def format_zephyr_step(step: dict) -> str:
    """
    step 구조체 → Zephyr '테스트 단계' 텍스트 변환
    *[Ver.]* + *[Precondition]* 마커 포함
    """
    lines = []
    ver = step.get("ver", "")
    spec_ref = step.get("spec_ref", "")
    preconditions = step.get("preconditions", [])

    if ver or spec_ref:
        lines.append("*[Ver.]*")
        if spec_ref:
            lines.append(spec_ref)
        elif ver:
            lines.append(f"ver.{ver}")
        lines.append("")

    if preconditions:
        lines.append("*[Precondition]*")
        for i, pre in enumerate(preconditions, 1):
            lines.append(f"{i}. {pre}")

    return "\n".join(lines).strip()


def format_zephyr_data(data: dict) -> str:
    """
    data 구조체 → Zephyr '데이터 테스트' 텍스트 변환
    *[Step]* 마커 포함
    """
    steps = data.get("steps", [])
    if not steps:
        return ""
    lines = ["*[Step]*"]
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def format_zephyr_result(result: dict) -> str:
    """
    result 구조체 → Zephyr '예상 결과' 텍스트 변환
    *[DUX]* + *[Action]* 마커 포함
    """
    lines = []
    dux = result.get("dux", [])
    actions = result.get("actions", [])

    if dux:
        lines.append("*[DUX]*")
        for i, elem in enumerate(dux, 1):
            lines.append(f"{i}. {elem}")

    if actions:
        if lines:
            lines.append("")
        lines.append("*[Action]*")
        for i, act in enumerate(actions, 1):
            if isinstance(act, dict):
                action_text = act.get("action", "")
                result_text = act.get("result", "")
                lines.append(f"{i}. {action_text}: {result_text}")
            else:
                # Zephyr 다운로드 데이터: 마커 없이 plain string으로 저장된 경우
                lines.append(f"{i}. {act}")

    return "\n".join(lines)


def req_to_zephyr_tc(req: dict) -> dict | None:
    """
    requirements 항목 → Zephyr TC 형식 변환

    두 가지 형식 모두 지원:
      A. 다중 스텝: tc.steps = [ {step, data, result}, ... ]   ← 권장
      B. 단일 스텝: tc.step / tc.data / tc.result              ← 하위 호환

    반환: {"summary": ..., "steps": [ {"step": ..., "data": ..., "result": ...}, ... ]}
    """
    tc_raw = req.get("tc")
    if not tc_raw:
        return None

    summary = tc_raw.get("summary", req.get("description", ""))
    zephyr_steps = []

    if "steps" in tc_raw:
        # A. 다중 스텝 형식 (권장)
        for s in tc_raw["steps"]:
            clean_s = strip_meta_keys(s)
            zephyr_steps.append({
                "step":   format_zephyr_step(clean_s.get("step", {})),
                "data":   format_zephyr_data(clean_s.get("data", {})),
                "result": format_zephyr_result(clean_s.get("result", {})),
            })
    else:
        # B. 단일 스텝 형식 (하위 호환)
        zephyr_steps.append({
            "step":   format_zephyr_step(tc_raw.get("step", {})),
            "data":   format_zephyr_data(tc_raw.get("data", {})),
            "result": format_zephyr_result(tc_raw.get("result", {})),
        })

    return {"summary": summary, "steps": zephyr_steps}


def strip_meta_keys(obj):
    """_로 시작하는 메타 키 제거 (재귀)"""
    if isinstance(obj, dict):
        return {k: strip_meta_keys(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_meta_keys(i) for i in obj]
    return obj


def convert_legacy_format(data: dict) -> dict | None:
    """
    tc_creation_examples 형식 → 표준 형식 변환
    {"requirement": "...", "tc": {...}}  →  {"spec": "...", "requirements": [...]}

    주의: legacy 형식에는 spec 원문이 없으므로 requirement를 spec으로 사용합니다.
    """
    if "requirement" not in data or "tc" not in data:
        return None

    tc = data["tc"]
    spec_text = f"[기능 요구사항]\n{data['requirement']}\n\n[관련 테스트 시나리오]\n{tc.get('title', '')}"

    # legacy → Zephyr TC 형식으로 변환
    steps = tc.get("steps", [])
    req = {
        "id": "REQ-001",
        "category": "기능",
        "description": data["requirement"],
        "priority": "medium",
        "testable": True,
        "tc": {
            "summary": tc.get("title", data["requirement"]),
            "step": {
                "ver": "",
                "spec_ref": "",
                "preconditions": tc.get("preconditions", [])
            },
            "data": {
                "steps": [s.get("action", "") for s in steps if s.get("action")]
            },
            "result": {
                "dux": [],
                "actions": [
                    {"action": s.get("action", ""), "result": s.get("expected", "")}
                    for s in steps if s.get("expected")
                ]
            }
        }
    }

    return {"spec": spec_text, "requirements": [req]}


def extract_relevant_spec(spec: str, requirements: list[dict], max_chars: int = 2000) -> str:
    """
    verbatim OCR 전체 텍스트에서 TC requirements와 관련된 구간만 추출.

    슬라이딩 윈도우로 키워드 밀도가 가장 높은 max_chars 구간을 찾고,
    단락 경계에서 잘라 반환한다.
    """
    if not spec or len(spec) <= max_chars:
        return spec

    # requirements descriptions에서 2자 이상 단어 추출
    keywords: set[str] = set()
    for req in requirements:
        desc = req.get("description", "")
        words = re.findall(r"[가-힣a-zA-Z]{2,}", desc)
        keywords.update(words)

    if not keywords:
        return spec[:max_chars]

    # 슬라이딩 윈도우 (100자 간격)로 키워드 밀도 최고 구간 탐색
    best_score = -1
    best_start = 0
    step = 100

    for start in range(0, max(1, len(spec) - max_chars + 1), step):
        window = spec[start:start + max_chars]
        score = sum(window.count(kw) for kw in keywords)
        if score > best_score:
            best_score = score
            best_start = start

    # 단락 경계에서 시작/끝 조정
    region = spec[best_start:best_start + max_chars]
    # 시작: 첫 번째 줄바꿈 이후부터 (앞쪽 잘린 단락 제거)
    if best_start > 0:
        nl = region.find("\n")
        if 0 < nl < 100:
            region = region[nl + 1:]
    # 끝: 마지막 줄바꿈까지 (뒤쪽 잘린 단락 제거)
    nl = region.rfind("\n")
    if nl > len(region) - 100:
        region = region[:nl]

    return region.strip() or spec[:max_chars]


def build_training_sample(spec: str, requirements: list[dict]) -> dict:
    """
    MLX-LM chat 형식 학습 샘플 생성

    모델 출력: requirements 목록 (Zephyr TC 포함)
    """
    # 메타 키 제거 후 직렬화
    clean_reqs = strip_meta_keys(requirements)

    # 학습 목표 출력: requirements + Zephyr 변환 결과
    output_reqs = []
    for req in clean_reqs:
        out = {k: v for k, v in req.items() if k != "tc"}
        zephyr_tc = req_to_zephyr_tc(req)
        if zephyr_tc:
            out["zephyr_tc"] = zephyr_tc
        output_reqs.append(out)

    req_output = {
        "requirements": output_reqs,
        "total_count": len(output_reqs),
        "testable_count": sum(1 for r in output_reqs if r.get("testable", True))
    }

    # extract_relevant_spec에서 이미 2000자로 제한됨
    spec_text = spec.strip() if spec and spec.strip() else ""

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    (
                        f"다음 기획서에서 테스트케이스를 작성해주세요.\n\n"
                        f"주의사항:\n"
                        f"- 기획서에 정의된 모든 항목을 누락 없이 포함하세요.\n"
                        f"- 기획서 내용으로 예상 가능한 예외 케이스(빈 상태, 긴 텍스트, 오류 상황 등)도 포함하세요.\n\n"
                        f"기획서:\n{spec_text}"
                    ) if spec_text else (
                        f"기획서 없이 요구사항 설명만으로 테스트케이스를 작성해주세요.\n\n"
                        f"주의사항:\n"
                        f"- 요구사항 description을 기반으로 테스트 시나리오를 도출하세요.\n"
                        f"- 예상 가능한 예외 케이스(빈 상태, 오류 상황 등)도 포함하세요."
                    )
                )
            },
            {
                "role": "assistant",
                "content": json.dumps(req_output, ensure_ascii=False, indent=2)
            }
        ]
    }


def mix_sliding_pairs(pairs: list[dict], max_combined_chars: int = 2500) -> list[dict]:
    """
    Mix 전략: 모든 단일 pair 유지 + 슬라이딩 윈도우 방식의 2-req 그룹 추가.

    - 단일 샘플: 모든 pair를 그대로 유지 (v8 수준 학습 품질 유지)
    - 슬라이딩 2-req: 같은 PDF의 인접 pair [i, i+1]을 묶어 추가
      - combined spec 길이 ≤ max_combined_chars인 경우만 허용 (시퀀스 잘림 방지)
    """
    import collections

    by_pdf = collections.defaultdict(list)
    for p in pairs:
        pdf_name = p.get("_pdf_name", "NO_PDF")
        if pdf_name != "NO_PDF":
            by_pdf[pdf_name].append(p)

    # 단일 샘플 전체 유지
    result = list(pairs)

    # 슬라이딩 2-req 추가
    added = 0
    for pdf_name, pdf_pairs in by_pdf.items():
        pdf_pairs_sorted = sorted(pdf_pairs, key=lambda p: p.get("_min_page", 0))
        for i in range(len(pdf_pairs_sorted) - 1):
            a, b = pdf_pairs_sorted[i], pdf_pairs_sorted[i + 1]
            combined_chars = len(a.get("spec", "")) + len(b.get("spec", ""))
            if combined_chars <= max_combined_chars:
                result.append({
                    "spec": a["spec"] + "\n\n" + b["spec"],
                    "requirements": a.get("requirements", []) + b.get("requirements", []),
                    "_pdf_name": pdf_name,
                })
                added += 1

    print(f"  Mix 결과: 단일 {len(pairs)}개 + 슬라이딩 2-req {added}개 = 총 {len(result)}개")
    return result


def group_pairs_by_pdf(pairs: list[dict], group_size: int) -> list[dict]:
    """
    같은 PDF에서 나온 pair들을 group_size 단위로 묶어 multi-requirement 샘플 생성.

    - pair['_pdf_name'] : PDF 파일명 (load_pairs에서 설정)
    - pair['_min_page'] : 첫 번째 페이지 번호 (정렬용)
    - 같은 PDF에서 페이지 순으로 정렬 후 group_size씩 청크
    - spec은 청크 내 spec들을 순서대로 이어 붙임
    - requirements는 청크 내 requirement들을 합침
    - PDF가 없거나 '_pdf_name' == 'NO_PDF'인 pair는 단독 샘플
    """
    import collections

    by_pdf = collections.defaultdict(list)
    no_pdf = []
    for p in pairs:
        pdf_name = p.get("_pdf_name", "NO_PDF")
        if pdf_name == "NO_PDF":
            no_pdf.append(p)
        else:
            by_pdf[pdf_name].append(p)

    grouped = []

    # 같은 PDF는 페이지 순 정렬 후 청크
    for pdf_name, pdf_pairs in by_pdf.items():
        pdf_pairs.sort(key=lambda p: p.get("_min_page", 0))
        for i in range(0, len(pdf_pairs), group_size):
            chunk = pdf_pairs[i:i + group_size]
            if len(chunk) == 1:
                grouped.append(chunk[0])
            else:
                combined_spec = "\n\n".join(p["spec"] for p in chunk if p.get("spec"))
                combined_reqs = []
                for p in chunk:
                    combined_reqs.extend(p.get("requirements", []))
                grouped.append({
                    "spec": combined_spec,
                    "requirements": combined_reqs,
                    "_pdf_name": pdf_name,
                })

    grouped.extend(no_pdf)
    print(f"  그룹핑 결과: {len(pairs)}개 pair → {len(grouped)}개 샘플 (group_size={group_size})")
    return grouped


def group_pairs_by_page(pairs: list[dict]) -> list[dict]:
    """같은 (PDF, 페이지)를 공유하는 pair들을 자연 단위로 묶어 multi-req 샘플 생성.

    - 같은 페이지의 pair들은 동일한 spec(같은 OCR)을 가지므로 spec은 1개만 사용
    - requirements는 해당 페이지의 모든 req를 합침
    - 페이지 정보 없는 pair(_page_key=None)는 단독 샘플
    - 추론(페이지 1개 → 그 페이지의 모든 TC)과 동일한 구조를 학습
    """
    import collections

    by_page: dict = collections.OrderedDict()
    singles = []
    for p in pairs:
        key = p.get("_page_key")
        if not key:
            singles.append(p)
        else:
            by_page.setdefault(key, []).append(p)

    grouped = []
    for key, page_pairs in by_page.items():
        spec = next((p["spec"] for p in page_pairs if p.get("spec")), "")
        combined_reqs = []
        for p in page_pairs:
            combined_reqs.extend(p.get("requirements", []))
        grouped.append({
            "spec": spec,
            "requirements": combined_reqs,
            "_pdf_name": key[0],
        })

    grouped.extend(singles)
    from collections import Counter
    dist = Counter(len(g["requirements"]) for g in grouped)
    print(f"  페이지 그룹핑: {len(pairs)}개 pair → {len(grouped)}개 샘플")
    print(f"  req 분포: {dict(sorted(dist.items()))}")
    return grouped


_PDF_FALLBACK_DIR: Path | None = None  # --pdf-dir 값, main()에서 설정


def load_pairs(input_dir: Path, vlm_ocr: bool = False) -> list[dict]:
    """디렉토리에서 (spec, requirements) 쌍 로드"""
    pairs = []
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"⚠️  {input_dir} 에 JSON 파일이 없습니다.")
        return []

    print(f"📂 JSON 파일 {len(json_files)}개 발견")

    for f in json_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))

            # 표준 형식 (spec 텍스트 직접 또는 spec_pdf 참조)
            if "requirements" in data and ("spec" in data or "spec_pdf" in data):

                # spec이 비어 있고 spec_pdf가 있으면 PDF에서 추출
                if not data.get("spec") and data.get("spec_pdf"):
                    pdf_info = data["spec_pdf"]
                    pdf_path = pdf_info.get("path", "")
                    pages    = pdf_info.get("pages")  # list[int] or None

                    # 상대 경로는 JSON 파일 위치 기준으로 해석
                    if pdf_path and not Path(pdf_path).is_absolute():
                        pdf_path = str(f.parent / pdf_path)

                    # 절대 경로가 현 머신에 없으면 --pdf-dir에서 파일명 기준 폴백
                    if pdf_path and not Path(pdf_path).exists() and _PDF_FALLBACK_DIR:
                        fallback = _PDF_FALLBACK_DIR / Path(pdf_path).name
                        if fallback.exists():
                            pdf_path = str(fallback)
                        else:
                            print(f"  ⚠️  {f.name}: PDF 없음 ({Path(pdf_path).name}) — 건너뜀")
                            pdf_path = ""

                    if pdf_path:
                        print(f"  📄 {f.name}: PDF에서 spec 추출 중...")
                        data["spec"] = load_spec_from_pdf(pdf_path, pages,
                                                          use_vlm_ocr=vlm_ocr)
                    else:
                        print(f"  ⚠️  {f.name}: spec_pdf.path가 비어 있습니다.")

                # spec이 없으면 spec 없는 샘플로 포함 (description만으로 TC 생성 학습)
                if not data.get("spec"):
                    data["spec"] = ""

                # (C) extract_relevant_spec 제거: 절충형 OCR로 spec이 짧고 정제되어
                #     추론(키워드 없음)과 동일하게 전체 spec을 그대로 사용 — 학습=추론 일치

                # PDF 그룹핑용 메타 정보 저장
                spec_pdf = data.get("spec_pdf", {})
                if isinstance(spec_pdf, dict):
                    pdf_path = spec_pdf.get("path", "")
                    data["_pdf_name"] = Path(pdf_path).name if pdf_path else "NO_PDF"
                    pages = spec_pdf.get("pages") or []
                    data["_min_page"] = min(pages) if pages else 0
                    data["_page_key"] = (data["_pdf_name"], tuple(pages))
                else:
                    data["_pdf_name"] = "NO_PDF"
                    data["_min_page"] = 0
                    data["_page_key"] = None

                pairs.append(data)
                print(f"  ✅ {f.name}: 요구사항 {len(data['requirements'])}개")

            # legacy 형식 (tc_creation_examples)
            elif "requirement" in data and "tc" in data:
                converted = convert_legacy_format(data)
                if converted:
                    pairs.append(converted)
                    print(f"  🔄 {f.name}: legacy 형식 변환 완료")

            else:
                print(f"  ⚠️  {f.name}: 인식할 수 없는 형식 (건너뜀)")

        except json.JSONDecodeError as e:
            print(f"  ❌ {f.name}: JSON 파싱 오류 - {e}")

    return pairs


def _estimate_tokens(text: str) -> int:
    return int(len(text) / 2.0)


def filter_by_seq_length(samples: list[dict], max_seq_length: int = 3072) -> list[dict]:
    """assistant 응답이 잘리는 샘플 제거"""
    filtered = []
    removed = 0
    for s in samples:
        msgs = s["messages"]
        sys_t  = _estimate_tokens(msgs[0]["content"])
        user_t = _estimate_tokens(msgs[1]["content"])
        asst_t = _estimate_tokens(msgs[2]["content"])
        remaining = max_seq_length - sys_t - user_t
        if remaining >= asst_t:
            filtered.append(s)
        else:
            removed += 1
    if removed:
        print(f"  🗑️  assistant 잘림 샘플 {removed}개 제거 (max_seq={max_seq_length})")
    return filtered


def split_and_save(samples: list[dict], output_dir: Path, valid_ratio: float = 0.15,
                   max_seq_length: int = 3072):
    """train / valid 분리 저장"""
    samples = filter_by_seq_length(samples, max_seq_length)
    random.shuffle(samples)

    n_valid = max(1, int(len(samples) * valid_ratio))
    valid_samples = samples[:n_valid]
    train_samples = samples[n_valid:]

    for name, data in [("train", train_samples), ("valid", valid_samples)]:
        out_path = output_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  💾 {out_path.name}: {len(data)}개 샘플")

    return len(train_samples), len(valid_samples)


def show_sample(samples: list[dict]):
    """샘플 하나 미리보기"""
    if not samples:
        return
    s = samples[0]
    print("\n" + "─" * 60)
    print("📋 샘플 미리보기 (첫 번째 항목)")
    print("─" * 60)
    for msg in s["messages"]:
        role = msg["role"].upper()
        content = msg["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"\n[{role}]\n{content}")
    print("─" * 60)


def main():
    parser = argparse.ArgumentParser(description="파인튜닝 학습 데이터 준비")
    parser.add_argument("--input-dir", default="./raw_data",
                        help="(spec, TC) 쌍 JSON 파일 디렉토리 (기본: ./raw_data)")
    parser.add_argument("--output-dir", default="./data",
                        help="JSONL 출력 디렉토리 (기본: ./data)")
    parser.add_argument("--valid-ratio", type=float, default=0.15,
                        help="검증 데이터 비율 (기본: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본: 42)")
    parser.add_argument("--use-examples", action="store_true",
                        help="tc_creation_examples/ 의 legacy 데이터도 포함")
    parser.add_argument("--vlm-ocr", action="store_true",
                        help="Qwen2.5-VL-7B로 PDF OCR (깨끗한 한글 추출, 느리지만 품질 높음)")
    parser.add_argument("--group-size", type=int, default=1,
                        help="같은 PDF에서 나온 pair를 N개씩 묶어 multi-requirement 샘플 생성 (기본: 1=비활성)")
    parser.add_argument("--mix-sliding", action="store_true",
                        help="단일 샘플 전체 유지 + 슬라이딩 2-req 그룹 추가 (구버전)")
    parser.add_argument("--by-page", action="store_true",
                        help="(PDF,페이지) 자연 단위로 묶어 페이지의 모든 TC를 한 샘플로 (v15 권장, 추론과 일치)")
    parser.add_argument("--max-combined-chars", type=int, default=2500,
                        help="--mix-sliding 시 허용할 combined spec 최대 길이 (기본: 2500)")
    parser.add_argument("--pdf-dir", default=None,
                        help="PDF 파일 검색 디렉토리 — raw_data의 절대경로가 현 머신에 없을 때 파일명 기준 폴백 탐색")
    args = parser.parse_args()

    global _PDF_FALLBACK_DIR
    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf_dir:
        _PDF_FALLBACK_DIR = Path(args.pdf_dir).expanduser().resolve()
        print(f"📁 PDF 폴백 디렉토리: {_PDF_FALLBACK_DIR}")

    print("=" * 60)
    print("📚 파인튜닝 데이터 준비")
    print("=" * 60)

    if args.vlm_ocr:
        print("🔭 VLM OCR 모드: Qwen2.5-VL-7B로 PDF 페이지 OCR (깨끗한 한글)")
        load_ocr_cache(output_dir)

    # 데이터 로드
    pairs = load_pairs(input_dir, vlm_ocr=args.vlm_ocr)

    # legacy 예시 데이터 추가 (--use-examples 옵션)
    if args.use_examples:
        examples_dir = Path(__file__).parent.parent / "tc_creation_examples"
        if examples_dir.exists():
            print(f"\n📂 legacy 예시 데이터 로드: {examples_dir}")
            legacy_pairs = load_pairs(examples_dir)
            pairs.extend(legacy_pairs)
            print(f"  → 총 {len(legacy_pairs)}개 추가")

    # (PDF,페이지) 자연 단위 그룹핑 (v15 권장)
    if args.by_page:
        print(f"\n📦 페이지 단위 그룹핑 모드 (추론과 동일 구조)")
        pairs = group_pairs_by_page(pairs)
    # Mix 슬라이딩 (구버전)
    elif args.mix_sliding:
        print(f"\n📦 Mix 슬라이딩 모드 (max_combined_chars={args.max_combined_chars})")
        pairs = mix_sliding_pairs(pairs, args.max_combined_chars)
    # PDF 기준 multi-requirement 그룹핑
    elif args.group_size > 1:
        print(f"\n📦 PDF 기준 그룹핑 (group_size={args.group_size})")
        pairs = group_pairs_by_pdf(pairs, args.group_size)

    if not pairs:
        print("\n❌ 로드된 데이터가 없습니다.")
        print("\n📌 raw_data/ 디렉토리에 다음 형식의 JSON 파일을 추가하세요:")
        print("""
  {
    "spec": "기획서 원문 텍스트...",
    "requirements": [
      {
        "id": "REQ-001",
        "category": "기능",
        "description": "요구사항 설명",
        "priority": "high",
        "acceptance_criteria": ["기준1", "기준2"]
      }
    ]
  }""")
        return

    # 학습 샘플 생성
    samples = []
    for pair in pairs:
        sample = build_training_sample(pair["spec"], pair["requirements"])
        samples.append(sample)

    print(f"\n✅ 총 {len(samples)}개 학습 샘플 생성")

    # 미리보기
    show_sample(samples)

    # 분리 저장
    print(f"\n📊 train/valid 분리 (valid {args.valid_ratio*100:.0f}%)")
    n_train, n_valid = split_and_save(samples, output_dir, args.valid_ratio)

    print(f"""
{'=' * 60}
✅ 데이터 준비 완료!
{'=' * 60}
  학습: {n_train}개
  검증: {n_valid}개
  출력: {output_dir.resolve()}

다음 단계:
  bash finetune.sh
{'=' * 60}""")


if __name__ == "__main__":
    main()
