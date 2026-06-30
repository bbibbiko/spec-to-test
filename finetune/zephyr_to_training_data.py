#!/usr/bin/env python3
"""
Zephyr TC → 파인튜닝 학습 데이터 변환 스크립트

Zephyr Squad에서 TC를 일괄 다운로드하여 파인튜닝 학습 데이터(raw_data/) 형식으로 변환합니다.

사용법:
  # 특정 키만 다운로드 (쉼표 구분 또는 여러 번 입력)
  python zephyr_to_training_data.py --keys QA-256,QA-258
  python zephyr_to_training_data.py --keys QA-256 --keys QA-258

  # PDF 기획서에서 spec 자동 추출 + TC 다운로드 통합
  python zephyr_to_training_data.py --keys QA-100,QA-101 --pdf spec.pdf
  python zephyr_to_training_data.py --keys QA-100,QA-101 --pdf spec.pdf --pdf-pages 2,3,5

  # 전체 / 일부
  python zephyr_to_training_data.py                     # 전체 TC 다운로드
  python zephyr_to_training_data.py --max-tc 50         # 최대 50개

  # 기타 옵션
  python zephyr_to_training_data.py --project MYAPP     # 다른 프로젝트
  python zephyr_to_training_data.py --overwrite         # 기존 파일 덮어쓰기
  python zephyr_to_training_data.py --skip-empty-steps  # 스텝 없는 TC 제외
  python zephyr_to_training_data.py --dry-run           # 저장 없이 미리보기

출력: finetune/raw_data/pair_{issueKey}.json
  --pdf 없으면: spec 필드는 "" (수동 입력 필요)
  --pdf 있으면: spec 필드에 Claude Vision이 추출한 기획서 내용 자동 삽입

PDF 의존성:
  pip install pymupdf anthropic
"""

import json
import time
import argparse
import re
import os
import base64
from pathlib import Path

import requests

# ─── 설정 ──────────────────────────────────────────────────────────────────────
JIRA_URL = "https://jira.example.com"
PAT_TOKEN = os.getenv("JIRA_PAT", "")   # 환경변수로 주입 (코드에 토큰 하드코딩 금지)
PROJECT_KEY = "QA"

# Claude Vision API (PDF spec 추출용)
# 환경변수 ANTHROPIC_API_KEY 또는 아래에 직접 입력
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VISION_MODEL = "claude-sonnet-4-6"   # Vision 지원 모델

HEADERS = {
    "Authorization": f"Bearer {PAT_TOKEN}",
    "Content-Type": "application/json"
}

REQUEST_DELAY = 0.3   # API 부하 방지 (초)
PAGE_SIZE = 100       # 한 번에 가져올 이슈 수


# ─── Zephyr 마커 파싱 ──────────────────────────────────────────────────────────

def _split_numbered_list(text: str) -> list[str]:
    """
    번호/불릿 목록 텍스트 → 리스트 변환

    지원 형식:
      "1. 항목A"  "2) 항목B"  "* 항목C"  "- 항목D"  "• 항목E"
    번호/마커 없는 줄도 그대로 포함
    """
    items = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 번호(1. 2) ...) 또는 불릿(* - • ▪) 제거
        cleaned = re.sub(r"^(\d+[\.\)]\s*|[*\-•▪▸]\s+)", "", line)
        if cleaned:
            items.append(cleaned)
    return items


def _extract_section(text: str, marker: str, stop_markers: list[str] = None) -> str:
    """마커 이후 섹션 텍스트 추출"""
    escaped = re.escape(marker)
    stop = "|".join(re.escape(m) for m in (stop_markers or [])) or r"\Z"
    pattern = rf"{escaped}\s*\n(.*?)(?={stop}|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_step_field(text: str) -> dict:
    """
    Zephyr '테스트 단계' → 구조화된 dict

    입력:
      *[Ver.]*
      spec.pdf p.2

      *[Precondition]*
      1. 앱 최초 설치
      2. 알림 권한 OFF

    출력:
      {"ver": "", "spec_ref": "spec.pdf p.2", "preconditions": ["앱 최초 설치", "알림 권한 OFF"]}
    """
    result = {"ver": "", "spec_ref": "", "preconditions": []}
    if not text:
        return result

    # *[Ver.]* 섹션
    if "*[Ver.]*" in text:
        ver_content = _extract_section(text, "*[Ver.]*", ["*[Precondition]*"])
        lines = [l.strip() for l in ver_content.split("\n") if l.strip()]
        if lines:
            line = lines[0]
            # 파일명 패턴이면 spec_ref, 버전 패턴이면 ver
            if re.search(r"\.(pdf|docx|xlsx|pptx|hwp)", line, re.I):
                result["spec_ref"] = line
            elif re.match(r"ver\.?\s*[\d.]+", line, re.I):
                result["ver"] = line
            else:
                result["spec_ref"] = line

    # *[Precondition]* 섹션
    if "*[Precondition]*" in text:
        pre_content = _extract_section(text, "*[Precondition]*")
        result["preconditions"] = _split_numbered_list(pre_content)

    # 마커 없이 순수 텍스트인 경우 → preconditions로 처리
    if not result["preconditions"] and not result["spec_ref"] and not result["ver"]:
        stripped = text.strip()
        if stripped:
            result["preconditions"] = _split_numbered_list(stripped) or [stripped]

    return result


def parse_data_field(text: str) -> dict:
    """
    Zephyr '데이터 테스트' → 구조화된 dict

    입력:
      *[Step]*
      1. 앱 실행
      2. 알림 권한 요청까지 진행

    출력:
      {"steps": ["앱 실행", "알림 권한 요청까지 진행"]}
    """
    if not text:
        return {"steps": []}

    if "*[Step]*" in text:
        content = _extract_section(text, "*[Step]*")
        return {"steps": _split_numbered_list(content)}
    else:
        # 마커 없는 경우 그대로 파싱
        return {"steps": _split_numbered_list(text)}


def parse_result_field(text: str) -> dict:
    """
    Zephyr '예상 결과' → 구조화된 dict

    입력:
      *[DUX]*
      1. OS 기본 알림 팝업
      2. '허용 안 함' 버튼

      *[Action]*
      1. '허용 안 함' 선택: OS 팝업 닫힘
      2. '허용' 선택: 알림 권한 허용됨

    출력:
      {
        "dux": ["OS 기본 알림 팝업", "'허용 안 함' 버튼"],
        "actions": [
          {"action": "'허용 안 함' 선택", "result": "OS 팝업 닫힘"},
          {"action": "'허용' 선택", "result": "알림 권한 허용됨"}
        ]
      }
    """
    result = {"dux": [], "actions": []}
    if not text:
        return result

    # *[DUX]* 섹션
    if "*[DUX]*" in text:
        dux_content = _extract_section(text, "*[DUX]*", ["*[Action]*"])
        result["dux"] = _split_numbered_list(dux_content)

    # *[Action]* 섹션
    if "*[Action]*" in text:
        action_content = _extract_section(text, "*[Action]*")
        for line in action_content.split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if not cleaned:
                continue
            # "액션: 결과" 형식 (첫 번째 ':' 기준 분리)
            if ":" in cleaned:
                action_part, _, result_part = cleaned.partition(":")
                result["actions"].append({
                    "action": action_part.strip(),
                    "result": result_part.strip()
                })
            else:
                result["actions"].append({"action": cleaned, "result": ""})

    # 마커 없이 순수 텍스트인 경우 → dux로 처리
    if not result["dux"] and not result["actions"] and text.strip():
        result["dux"] = _split_numbered_list(text) or [text.strip()]

    return result


def parse_zephyr_steps(raw_steps: list) -> list:
    """
    Zephyr API 응답 steps → template.json steps 배열 변환

    입력: [{"step": "...", "data": "...", "result": "..."}, ...]
    출력: [{"step": {...}, "data": {...}, "result": {...}}, ...]
    """
    parsed = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        parsed.append({
            "step":   parse_step_field(s.get("step", "") or ""),
            "data":   parse_data_field(s.get("data", "") or ""),
            "result": parse_result_field(s.get("result", "") or ""),
        })
    return parsed


# ─── PDF → spec 추출 (Claude Vision) ──────────────────────────────────────────

VISION_PROMPT = """이 기획서 페이지의 내용을 모바일 앱 테스트 케이스 작성에 필요한 형태로 정리해주세요.

다음 내용을 빠짐없이 포함해주세요:
- 화면에 표시되는 UI 요소 (버튼, 텍스트, 아이콘, 팝업 등)
- 기능 동작 조건 및 분기 (ON/OFF, 권한 여부, 로그인 상태 등)
- 사용자 액션에 따른 동작 흐름
- 에러 처리 / 예외 상황
- 버전 정보, 정책 조건이 있으면 그대로 포함

이미지나 목업이 있으면 화면 구성 요소를 텍스트로 상세히 설명해주세요.
원문 표현을 최대한 유지하고, 요약하지 마세요."""


def _check_pdf_deps() -> bool:
    """pymupdf, anthropic 설치 여부 확인"""
    missing = []
    try:
        import fitz  # noqa
    except ImportError:
        missing.append("pymupdf")
    try:
        import anthropic  # noqa
    except ImportError:
        missing.append("anthropic")
    if missing:
        print(f"  ❌ PDF 추출에 필요한 패키지가 없습니다: {', '.join(missing)}")
        print(f"     pip install {' '.join(missing)}")
        return False
    return True


def _pdf_pages_to_images(pdf_path: str, page_nums: list[int] | None) -> list[tuple[int, bytes]]:
    """
    PDF 페이지 → PNG 바이트 변환

    page_nums: 1-base 페이지 번호 목록 (None이면 전체)
    반환: [(페이지번호, png_bytes), ...]
    """
    import fitz

    doc = fitz.open(pdf_path)
    total = len(doc)

    if page_nums is None:
        targets = list(range(1, total + 1))
    else:
        targets = [p for p in page_nums if 1 <= p <= total]
        out_of_range = [p for p in page_nums if p < 1 or p > total]
        if out_of_range:
            print(f"  ⚠️  PDF 범위 초과 페이지 무시: {out_of_range} (총 {total}페이지)")

    result = []
    for page_num in targets:
        page = doc[page_num - 1]          # 0-base
        # 2x 해상도로 렌더링 (이미지 품질 향상)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        result.append((page_num, pix.tobytes("png")))

    doc.close()
    return result


def _extract_page_spec(image_bytes: bytes, page_num: int, api_key: str) -> str:
    """
    Claude Vision으로 단일 페이지 이미지 → spec 텍스트 추출
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    message = client.messages.create(
        model=VISION_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": VISION_PROMPT,
                    },
                ],
            }
        ],
    )

    return message.content[0].text.strip()


def extract_spec_from_pdf(
    pdf_path: str,
    page_nums: list[int] | None = None,
    api_key: str = "",
) -> str:
    """
    PDF 파일에서 spec 텍스트 추출

    Args:
        pdf_path : PDF 파일 경로
        page_nums: 추출할 페이지 번호 목록 (1-base, None=전체)
        api_key  : Anthropic API key

    Returns:
        pages를 합친 spec 텍스트 (섹션 구분 포함)
    """
    if not _check_pdf_deps():
        return ""

    if not api_key:
        print("  ❌ ANTHROPIC_API_KEY가 없습니다.")
        print("     export ANTHROPIC_API_KEY=sk-ant-... 또는 스크립트 상단에 직접 입력")
        return ""

    pdf_path = str(Path(pdf_path).expanduser().resolve())
    if not Path(pdf_path).exists():
        print(f"  ❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return ""

    print(f"\n📄 PDF 기획서 분석 중: {Path(pdf_path).name}")

    pages = _pdf_pages_to_images(pdf_path, page_nums)
    if not pages:
        print("  ⚠️  추출할 페이지가 없습니다.")
        return ""

    page_label = f"{pages[0][0]}~{pages[-1][0]}p" if len(pages) > 1 else f"{pages[0][0]}p"
    print(f"  🖼️  {len(pages)}개 페이지 ({page_label}) → Claude Vision 분석...")

    spec_parts = []
    for i, (page_num, img_bytes) in enumerate(pages, 1):
        print(f"  [{i}/{len(pages)}] p.{page_num} 분석 중...", end="", flush=True)
        try:
            text = _extract_page_spec(img_bytes, page_num, api_key)
            spec_parts.append(f"[p.{page_num}]\n{text}")
            print(f" ✅ ({len(text)}자)")
        except Exception as e:
            print(f" ❌ 오류: {e}")
            spec_parts.append(f"[p.{page_num}]\n(추출 실패: {e})")
        time.sleep(0.5)   # API rate limit 방지

    combined = "\n\n".join(spec_parts)
    print(f"  ✅ 총 {len(combined)}자 추출 완료\n")
    return combined


# ─── Jira / Zephyr API ─────────────────────────────────────────────────────────

def get_all_test_issues(project_key: str, max_results: int = None) -> list:
    """
    JQL로 프로젝트의 모든 'Test' 이슈 조회 (페이지네이션 자동 처리)

    반환:
      [{"id": "12345", "key": "QA-123", "summary": "...",
        "labels": [...], "components": [...], "priority": "high"}, ...]
    """
    url = f"{JIRA_URL}/rest/api/2/search"
    all_issues = []
    start_at = 0

    while True:
        params = {
            "jql": f"project = '{project_key}' AND issuetype = 'Test Case' ORDER BY key ASC",
            "startAt": start_at,
            "maxResults": PAGE_SIZE,
            "fields": "summary,labels,components,priority,status"
        }

        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Jira 연결 실패: {JIRA_URL}")
            break
        except requests.exceptions.Timeout:
            print("  ❌ 요청 타임아웃")
            break

        if not resp.ok:
            print(f"  ❌ JQL 조회 실패: {resp.status_code}")
            print(f"     응답: {resp.text[:300]}")
            break

        data = resp.json()
        issues = data.get("issues", [])
        total = data.get("total", 0)

        if start_at == 0:
            print(f"  📊 총 {total}개 TC 발견")

        for issue in issues:
            fields = issue.get("fields", {})
            priority_raw = (fields.get("priority") or {}).get("name", "medium").lower()
            priority = priority_raw if priority_raw in ("high", "medium", "low") else "medium"

            all_issues.append({
                "id": issue.get("id"),
                "key": issue.get("key"),
                "summary": fields.get("summary", ""),
                "labels": fields.get("labels", []),
                "components": [c.get("name", "") for c in fields.get("components", [])],
                "priority": priority,
            })

        start_at += len(issues)
        print(f"  📥 {min(start_at, total)}/{total} 로드 완료", end="\r")

        if start_at >= total or not issues:
            break
        if max_results and len(all_issues) >= max_results:
            break

        time.sleep(REQUEST_DELAY)

    print()  # \r 줄 정리
    return all_issues[:max_results] if max_results else all_issues


def get_issues_by_keys(keys: list[str]) -> list:
    """
    이슈 키 목록으로 직접 조회
    JQL: key in (QA-256, QA-258, ...)

    반환: get_all_test_issues()와 동일한 형식
    """
    if not keys:
        return []

    # 쉼표로 구분된 키 목록 정규화 (공백, 대소문자)
    normalized = [k.strip().upper() for k in keys if k.strip()]
    if not normalized:
        return []

    key_list = ", ".join(normalized)
    url = f"{JIRA_URL}/rest/api/2/search"
    params = {
        "jql": f"key in ({key_list}) ORDER BY key ASC",
        "maxResults": len(normalized),
        "fields": "summary,labels,components,priority,status,issuetype"
    }

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Jira 연결 실패: {JIRA_URL}")
        return []

    if not resp.ok:
        print(f"  ❌ 조회 실패: {resp.status_code}")
        print(f"     응답: {resp.text[:300]}")
        return []

    issues_raw = resp.json().get("issues", [])
    result = []
    for issue in issues_raw:
        fields = issue.get("fields", {})
        priority_raw = (fields.get("priority") or {}).get("name", "medium").lower()
        priority = priority_raw if priority_raw in ("high", "medium", "low") else "medium"
        result.append({
            "id": issue.get("id"),
            "key": issue.get("key"),
            "summary": fields.get("summary", ""),
            "labels": fields.get("labels", []),
            "components": [c.get("name", "") for c in fields.get("components", [])],
            "priority": priority,
        })

    # 입력 순서와 다를 수 있으므로 입력 키 순서대로 재정렬
    key_order = {k: i for i, k in enumerate(normalized)}
    result.sort(key=lambda x: key_order.get(x["key"], 9999))

    # 찾지 못한 키 경고
    found_keys = {r["key"] for r in result}
    missing = [k for k in normalized if k not in found_keys]
    if missing:
        print(f"  ⚠️  찾을 수 없는 키: {', '.join(missing)}")

    return result


def get_test_steps_api(issue_id: str) -> list:
    """
    Zephyr API로 TC 스텝 조회
    GET /rest/zapi/latest/teststep/{issue_id}
    """
    url = f"{JIRA_URL}/rest/zapi/latest/teststep/{issue_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
    except Exception:
        return []

    if not resp.ok:
        return []

    data = resp.json()
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("stepBeanCollection", data.get("steps", []))
    return []


# ─── 변환 ──────────────────────────────────────────────────────────────────────

def issue_to_pair(issue: dict, steps: list,
                  spec: str = "",
                  spec_pdf_path: str = "",
                  spec_pdf_pages: list[int] | None = None) -> dict:
    """
    Jira 이슈 + Zephyr 스텝 → raw_data JSON 구조 생성

    spec          : 기획서 텍스트 직접 입력 (있으면 우선)
    spec_pdf_path : PDF 경로 참조 (prepare_data.py가 학습 시 읽음)
    spec_pdf_pages: 추출할 페이지 목록 (1-base, None=전체)
    """
    issue_key = issue.get("key", "")
    summary = issue.get("summary", "")
    priority = issue.get("priority", "medium")

    components = issue.get("components", [])
    labels = issue.get("labels", [])
    category = components[0] if components else (labels[0] if labels else "기능")

    parsed_steps = parse_zephyr_steps(steps)

    requirement = {
        "id": issue_key,
        "category": category,
        "description": summary,
        "testable": True,
        "priority": priority,
        "tc": {
            "summary": summary,
            "steps": parsed_steps
        }
    }

    pair: dict = {
        "_comment": f"Zephyr에서 자동 변환 ({issue_key}).",
        "spec": spec,
        "requirements": [requirement]
    }

    # PDF 참조 저장 (spec이 없을 때만)
    if spec_pdf_path and not spec:
        pair["spec_pdf"] = {"path": spec_pdf_path}
        if spec_pdf_pages:
            pair["spec_pdf"]["pages"] = spec_pdf_pages
        pair["_comment"] += f" spec_pdf 참조 저장 → prepare_data.py 실행 시 자동 추출."
    elif not spec and not spec_pdf_path:
        pair["_comment"] += " 'spec' 또는 'spec_pdf' 필드를 채워주세요."

    return pair


# ─── 메인 ──────────────────────────────────────────────────────────────────────

def _parse_keys_arg(keys_args: list[str]) -> list[str]:
    """
    --keys 인자 파싱: 쉼표 구분 또는 여러 번 입력 모두 지원
      --keys QA-256,QA-258
      --keys QA-256 --keys QA-258
    """
    result = []
    for arg in keys_args:
        for key in arg.split(","):
            key = key.strip()
            if key:
                result.append(key)
    return result


def _parse_pages_arg(pages_str: str | None) -> list[int] | None:
    """
    --pdf-pages 인자 파싱: "2,3,5-7,10" → [2, 3, 5, 6, 7, 10]
    None이면 전체 페이지
    """
    if not pages_str:
        return None
    result = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, _, end = part.partition("-")
            try:
                result.extend(range(int(start), int(end) + 1))
            except ValueError:
                print(f"  ⚠️  페이지 범위 파싱 실패: '{part}' (건너뜀)")
        else:
            try:
                result.append(int(part))
            except ValueError:
                print(f"  ⚠️  페이지 번호 파싱 실패: '{part}' (건너뜀)")
    return sorted(set(result)) if result else None


def main():
    parser = argparse.ArgumentParser(
        description="Zephyr TC를 파인튜닝 학습 데이터(raw_data/*.json)로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # TC 다운로드 + 기획서 PDF 경로 저장 (prepare_data.py 실행 시 자동 추출)
  python zephyr_to_training_data.py --keys QA-100,QA-101 \\
      --pdf specs/notification.pdf --pdf-pages 2,3,5-7

  # PDF 없이 TC만 다운로드 (spec은 나중에 수동 입력)
  python zephyr_to_training_data.py --keys QA-256,QA-258

  python zephyr_to_training_data.py --max-tc 30         # 처음 30개
  python zephyr_to_training_data.py                      # 전체 다운로드
  python zephyr_to_training_data.py --dry-run --keys QA-500
        """
    )
    parser.add_argument("--keys", action="append", default=[],
                        metavar="KEY[,KEY...]",
                        help="다운로드할 TC 키 (쉼표 구분 또는 반복 사용).")
    parser.add_argument("--project", default=PROJECT_KEY,
                        help=f"Jira 프로젝트 키 (기본: {PROJECT_KEY})")
    parser.add_argument("--output-dir",
                        default=str(Path(__file__).parent / "raw_data"),
                        help="출력 디렉토리 (기본: ./raw_data)")
    parser.add_argument("--max-tc", type=int, default=None,
                        help="최대 다운로드 TC 수 — --keys 없을 때만 적용 (기본: 전체)")
    parser.add_argument("--skip-empty-steps", action="store_true",
                        help="스텝이 없는 TC 건너뛰기")
    parser.add_argument("--overwrite", action="store_true",
                        help="기존 파일 덮어쓰기 (기본: 건너뜀)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 저장 없이 파싱 결과만 미리보기")
    # PDF 기획서 연결
    parser.add_argument("--pdf", metavar="PDF_PATH",
                        help="기획서 PDF 경로. 저장만 하고 prepare_data.py 실행 시 자동 추출")
    parser.add_argument("--pdf-pages", metavar="PAGES",
                        help="사용할 PDF 페이지. 예: '2,3,5-7,10'")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_keys = _parse_keys_arg(args.keys)

    # PDF 경로를 output_dir 기준 상대 경로로 변환 (이동해도 작동하도록)
    pdf_path_str = ""
    pdf_pages = None
    if args.pdf:
        abs_pdf = Path(args.pdf).expanduser().resolve()
        if abs_pdf.exists():
            # raw_data 디렉토리 기준 상대 경로로 저장
            try:
                pdf_path_str = str(abs_pdf.relative_to(output_dir))
            except ValueError:
                pdf_path_str = str(abs_pdf)  # 다른 드라이브면 절대 경로 유지
            pdf_pages = _parse_pages_arg(args.pdf_pages)
        else:
            print(f"⚠️  PDF 파일을 찾을 수 없습니다: {abs_pdf}")

    print("=" * 60)
    print("🔽 Zephyr TC → 파인튜닝 학습 데이터 변환")
    print("=" * 60)
    print(f"  프로젝트 : {args.project}")
    print(f"  출력 경로 : {output_dir.resolve()}")
    if target_keys:
        print(f"  대상 키  : {', '.join(target_keys)}")
    elif args.max_tc:
        print(f"  최대 TC수 : {args.max_tc}개")
    else:
        print(f"  대상     : 전체")
    if pdf_path_str:
        pages_label = args.pdf_pages or "전체"
        print(f"  PDF 기획서: {Path(args.pdf).name} (pages: {pages_label}) → 경로 저장")
    if args.dry_run:
        print("  모드      : dry-run (저장 없이 미리보기)")
    print()

    # ── 1. TC 목록 조회 ──────────────────────────────────────────────────────
    if target_keys:
        # 키 지정 모드
        print(f"📋 {len(target_keys)}개 키 조회 중...")
        issues = get_issues_by_keys(target_keys)
    else:
        # 전체 / max-tc 모드
        print("📋 TC 목록 조회 중...")
        issues = get_all_test_issues(args.project, args.max_tc)

    if not issues:
        print("\n❌ TC를 찾을 수 없습니다.")
        if target_keys:
            print(f"   입력한 키: {', '.join(target_keys)}")
        else:
            print("   - 프로젝트 키가 올바른지 확인하세요.")
            print("   - Jira 연결 / 권한을 확인하세요.")
        return

    print(f"✅ {len(issues)}개 TC 로드 완료\n")

    # ── 2. 스텝 다운로드 및 변환 ─────────────────────────────────────────────
    print("🔄 스텝 다운로드 및 변환 중...")
    print("-" * 60)

    saved = skipped = errors = 0
    # 키 지정 모드면 전부 미리보기, 전체 모드면 3개만
    preview_limit = len(issues) if target_keys else 3
    preview_count = 0

    for i, issue in enumerate(issues, 1):
        issue_key = issue["key"]
        issue_id  = issue["id"]
        summary   = issue["summary"]
        target    = output_dir / f"pair_{issue_key}.json"

        prefix = f"  [{i:>4}/{len(issues)}] {issue_key}"

        # 기존 파일 건너뛰기
        if not args.overwrite and not args.dry_run and target.exists():
            print(f"{prefix} — 이미 존재, 건너뜀")
            skipped += 1
            continue

        print(f"{prefix}: {summary[:35]}{'...' if len(summary) > 35 else ''}", end="")

        try:
            raw_steps = get_test_steps_api(issue_id)

            if args.skip_empty_steps and not raw_steps:
                print(" → 스텝 없음, 건너뜀")
                skipped += 1
                continue

            pair = issue_to_pair(issue, raw_steps,
                                 spec_pdf_path=pdf_path_str,
                                 spec_pdf_pages=pdf_pages)
            step_count = len(pair["requirements"][0]["tc"]["steps"])

            if args.dry_run:
                print(f" → {step_count}개 스텝")
                if preview_count < preview_limit:
                    print(f"\n{'─' * 50}")
                    print(f"[미리보기] {issue_key}")
                    print(json.dumps(pair, ensure_ascii=False, indent=2))
                    print(f"{'─' * 50}\n")
                preview_count += 1
            else:
                with open(target, "w", encoding="utf-8") as f:
                    json.dump(pair, f, ensure_ascii=False, indent=2)
                print(f" → ✅ {step_count}개 스텝 저장")
                saved += 1

        except Exception as e:
            print(f" → ❌ 오류: {e}")
            errors += 1

        time.sleep(REQUEST_DELAY)

    # ── 3. 결과 요약 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if args.dry_run:
        print("✅ dry-run 완료 (파일 저장 없음)")
    else:
        print("✅ 변환 완료!")
    print("=" * 60)
    if not args.dry_run:
        print(f"  저장    : {saved}개")
        print(f"  건너뜀  : {skipped}개")
        if errors:
            print(f"  오류    : {errors}개")
        print(f"  출력    : {output_dir.resolve()}")

    if pdf_path_str:
        next_steps = f"""
📌 다음 단계:
  1. spec_pdf 경로가 저장됐습니다 → prepare_data.py 실행 시 자동 추출됩니다.
     python prepare_data.py --input-dir ./raw_data

  2. 이미지 PDF라면 ANTHROPIC_API_KEY를 설정하세요 (텍스트 PDF는 불필요):
     export ANTHROPIC_API_KEY=sk-ant-..."""
    else:
        next_steps = """
📌 다음 단계:
  1. raw_data/pair_*.json 파일에 기획서를 연결하세요.
     방법 A) "spec" 필드에 텍스트 직접 붙여넣기
     방법 B) "spec_pdf" 필드로 PDF 참조:
       "spec_pdf": {{"path": "../../specs/foo.pdf", "pages": [2, 3, 4]}}

  2. 연결 완료 후:
     python prepare_data.py --input-dir ./raw_data"""

    print(next_steps)
    print("=" * 60)


if __name__ == "__main__":
    main()
