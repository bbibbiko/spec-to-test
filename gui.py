#!/usr/bin/env python3
"""
TC 생성 / 테스트 코드 생성 GUI
"""

import gradio as gr
import json
import os
import sys
import tempfile
import re
from pathlib import Path
from jira_client import JiraClient

# 파인튜닝 모듈 경로 등록
_TOOL_DIR = Path(__file__).parent
sys.path.insert(0, str(_TOOL_DIR / "code_gen"))
sys.path.insert(0, str(_TOOL_DIR / "finetune"))


def _try_recover_json(raw_str: str):
    """잘리거나 마크다운 블록으로 감싸진 JSON 문자열을 파싱 시도."""
    # 마크다운 코드 블록 제거
    text = re.sub(r'^```(?:json)?\s*', '', raw_str.strip())
    text = re.sub(r'\s*```\s*$', '', text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        return None
    candidate = text[start:end + 1] if end != -1 else text[start:]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 잘린 JSON 복구: 열린 괄호만큼 닫기
    truncated = candidate.rstrip(",\n ")
    open_b = truncated.count("{") - truncated.count("}")
    open_k = truncated.count("[") - truncated.count("]")
    closing = "]" * max(open_k, 0) + "}" * max(open_b, 0)
    try:
        return json.loads(truncated + closing)
    except json.JSONDecodeError:
        return None


def _dedup_steps(steps: list) -> list:
    """연속 중복 스텝 제거 (모델 반복 루프 결과물 정리)."""
    seen = []
    for s in steps:
        key = json.dumps(s, ensure_ascii=False, sort_keys=True)
        if not seen or seen[-1] != key:
            seen.append(key)
    return [json.loads(k) for k in seen]


def _format_tc_display(result: dict, pdf_name: str, page_range: str) -> str:
    """TC JSON → 사람이 읽기 쉬운 마크다운 (summary + steps만 표시)"""
    page_info = f"  |  **페이지**: {page_range}" if page_range else ""
    header = f"# ✅ TC 생성 완료\n\n> **PDF**: {pdf_name}{page_info}\n\n---\n\n"

    if not isinstance(result, dict):
        return header + str(result)

    # {"raw": "..."} 케이스: JSON 파싱 실패 → 복구 시도
    raw_str = result.get("raw")
    if raw_str and not result.get("requirements"):
        recovered = _try_recover_json(raw_str)
        if recovered and isinstance(recovered, dict):
            result = recovered
        else:
            # 복구 불가 → raw 문자열에서 읽을 수 있는 텍스트만 추출하여 표시
            # 마크다운 코드 블록·특수문자 제거 후 plain text 표시
            clean = re.sub(r'^```(?:json)?\s*', '', raw_str.strip())
            clean = re.sub(r'\s*```\s*$', '', clean)
            clean = re.sub(r'\\n', '\n', clean)
            clean = re.sub(r'\\"', '"', clean)
            return (
                header
                + "> ⚠️ JSON 파싱 부분 실패 (모델 출력이 잘렸거나 반복 루프 발생)\n\n"
                + "---\n\n"
                + clean
            )

    reqs = result.get("requirements", [])
    if not reqs:
        reqs = [result] if "summary" in result or "tc" in result or "zephyr_tc" in result else []

    if not reqs:
        return header + f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```"

    lines = [header]
    for i, req in enumerate(reqs, 1):
        tc = req.get("tc") or req.get("zephyr_tc") or req
        summary = tc.get("summary") or req.get("description", f"TC {i}")
        steps = _dedup_steps(tc.get("steps", []))

        lines.append(f"## {i}. {summary}\n")

        category = req.get("category", "")
        priority = req.get("priority", "")
        meta = " | ".join(filter(None, [category, f"우선순위: {priority}" if priority else ""]))
        if meta:
            lines.append(f"> {meta}\n")

        if steps:
            lines.append(f"### 테스트 스텝 ({len(steps)}개)\n")
            for j, s in enumerate(steps, 1):
                lines.append(f"**Step {j}**")

                step_val = s.get("step", "")
                if isinstance(step_val, dict):
                    precond = step_val.get("preconditions") or step_val.get("precondition", [])
                    if isinstance(precond, list):
                        precond = " / ".join(str(p) for p in precond if p)
                    if precond:
                        lines.append(f"- **사전조건**: {precond}")
                elif step_val:
                    lines.append(f"- **동작**: {step_val}")

                data_val = s.get("data", "")
                if isinstance(data_val, dict):
                    data_steps = data_val.get("steps", [])
                    if data_steps:
                        lines.append(f"- **입력**: {' / '.join(str(d) for d in data_steps)}")
                elif data_val:
                    lines.append(f"- **데이터**: {data_val}")

                res_val = s.get("result", "")
                if isinstance(res_val, dict):
                    action = res_val.get("action") or res_val.get("dux", "")
                    if isinstance(action, list):
                        action = " / ".join(str(a) for a in action if a)
                    if action:
                        lines.append(f"- **기대결과**: {action}")
                elif res_val:
                    lines.append(f"- **기대결과**: {res_val}")

                lines.append("")
        else:
            lines.append("*(스텝 없음)*\n")

        lines.append("---\n")

    return "\n".join(lines)


def _ocr_page_lengths(spec_text: str) -> dict:
    """OCR 텍스트를 [p.N] 마커로 나눠 페이지별 글자수 반환 ({'N': 글자수})."""
    lens = {}
    matches = list(re.finditer(r"\[p\.(\d+)\]\s*", spec_text or ""))
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(spec_text)
        lens[m.group(1)] = len(spec_text[start:end].strip())
    return lens


def _paginate_tc(result: dict, pdf_name: str, page_range: str, ocr_lens: dict = None):
    """결과를 페이지(req.page)별로 그룹핑 → ({라벨: 마크다운}, [라벨...]).

    페이지가 1개거나 page 정보가 없으면 단일 그룹(전체)으로 반환.
    ocr_lens가 있으면 각 페이지에 OCR 추출 글자수를 함께 표시(부실 진단용).
    """
    from collections import OrderedDict
    ocr_lens = ocr_lens or {}
    model_diag = (result.get("_diag") or {}) if isinstance(result, dict) else {}
    reqs = result.get("requirements") if isinstance(result, dict) else None
    if not reqs:
        # raw/단일 결과 — 페이지 구분 없이 전체 1장
        return {"전체": _format_tc_display(result, pdf_name, page_range)}, ["전체"]

    groups: "OrderedDict[str, list]" = OrderedDict()
    for req in reqs:
        pg = str(req.get("page", "") or "").strip()
        label = f"p.{pg}" if pg else "전체"
        groups.setdefault(label, []).append(req)

    pages, labels = {}, []
    for label, rs in groups.items():
        labels.append(label)
        pg = label[2:] if label.startswith("p.") else page_range
        md = _format_tc_display({"requirements": rs}, pdf_name, pg)
        # 진단: 모델 원시 case → 최종 case, OCR 추출량
        #   OCR 많은데 case 적음 → 모델 빈약 / 모델 원시 많은데 최종 적음 → 조립·dedup 손실
        pgk = pg.strip() if isinstance(pg, str) else ""
        n_cases = sum(len(r.get("zephyr_tc", {}).get("steps", [])) for r in rs)
        d = model_diag.get(pgk, {})
        diag = "\n\n> 🔎 진단 · "
        if d.get("raw_cases") is not None:
            diag += f"모델 생성 {d['raw_cases']}개 → 최종 {n_cases}개"
        else:
            diag += f"케이스 {n_cases}개"
        ocr_len = ocr_lens.get(pgk)
        if ocr_len is not None:
            diag += f" · OCR 추출 {ocr_len}자"
        md = md.replace("\n\n---\n\n", diag + "\n\n---\n\n", 1)
        pages[label] = md
    return pages, labels


def generate_tc_mlx_ui(
    pdf_file,
    page_range: str,
    progress=gr.Progress()
):
    """파인튜닝 EXAONE LoRA로 PDF → Zephyr TC 생성.

    출력 4개: (결과 마크다운, 다운로드 파일, 페이지 상태(dict), 페이지 선택 라디오 update).
    """
    if pdf_file is None:
        yield "# ❌ 오류\n\nPDF 파일을 업로드해주세요.", gr.update(), None, gr.update()
        return

    # Gradio가 동일 파일명에 대해 temp 경로를 재사용할 수 있으므로
    # 업로드된 파일을 타임스탬프가 붙은 고유 경로로 복사해서 사용한다.
    import shutil
    import datetime as _dt
    _raw_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    _orig_name = os.path.basename(_raw_path)
    pdf_path = os.path.join(tempfile.gettempdir(), f"tc_upload_{_ts}_{_orig_name}")
    shutil.copy2(_raw_path, pdf_path)

    pages = []
    if page_range and page_range.strip():
        import re as _re
        for part in page_range.split(","):
            part = part.strip()
            m = _re.match(r"(\d+)-(\d+)", part)
            if m:
                pages.extend(range(int(m.group(1)), int(m.group(2)) + 1))
            elif part.isdigit():
                pages.append(int(part))

    try:
        progress(0.1, desc="VLM OCR 준비 중...")
        yield "# 🔄 TC 생성 (EXAONE LoRA)\n\n**VLM OCR 중... (페이지당 ~1분, 학습과 동일 형식)**", gr.update(), None, gr.update()

        # 매 호출마다 모듈을 강제 재로드하여 모델 상태를 완전히 초기화
        import importlib, sys as _sys
        if "infer_pipeline" in _sys.modules:
            importlib.reload(_sys.modules["infer_pipeline"])
        # VLM OCR(ocr_pdf_pages) 사용 — 학습 데이터가 VLM OCR 형식이라 반드시 일치시켜야 함.
        # pymupdf(extract_pdf_pages)는 형식 불일치로 v20 어댑터 품질 저하(train/infer mismatch).
        from infer_pipeline import ocr_pdf_pages, generate_tc

        if not pages:
            import fitz
            doc = fitz.open(pdf_path)
            pages = list(range(1, len(doc) + 1))
            doc.close()

        progress(0.2, desc="VLM OCR 중 (페이지당 ~1분)...")
        spec_text = ocr_pdf_pages(pdf_path, pages)
        ocr_lens = _ocr_page_lengths(spec_text)

        progress(0.4, desc="EXAONE LoRA로 TC 생성 중...")
        yield (
            "# 🔄 TC 생성 (EXAONE LoRA)\n\n"
            "**✓ 텍스트 추출 완료**\n\n"
            "**⏳ EXAONE 3.5 7.8B + LoRA로 TC 생성 중...**\n\n"
            "*1-3분 소요*"
        ), gr.update(), None, gr.update()

        # pdf_name·page 전달 → [Ver.]에 "기획서명 - Np" 들어감 (CLI와 동일)
        result = generate_tc(spec_text, pdf_name=_orig_name,
                             page=",".join(map(str, pages)))

        progress(0.9, desc="결과 정리 중...")

        pages, labels = _paginate_tc(result, _orig_name, page_range, ocr_lens)
        # 업로드 JSON에는 진단 메타(_diag) 제외
        upload_result = {k: v for k, v in result.items() if k != "_diag"} \
            if isinstance(result, dict) else result
        result_text = json.dumps(upload_result, ensure_ascii=False, indent=2) \
            if isinstance(upload_result, dict) else str(upload_result)

        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_tc_mlx_{ts}.json", delete=False, encoding="utf-8"
        ) as f:
            f.write(result_text)
            out_path = f.name

        progress(1.0, desc="완료!")
        # 페이지가 2개 이상이면 페이지 선택 라디오 노출, 첫 페이지부터 표시
        first_label = labels[0]
        selector = gr.update(
            choices=labels, value=first_label, visible=len(labels) > 1,
            label=f"📑 페이지 ({len(labels)}장)"
        )
        yield pages[first_label], out_path, pages, selector

    except Exception as e:
        import traceback
        yield (f"# ❌ 오류\n\n```\n{e}\n\n{traceback.format_exc()}\n```",
               gr.update(), None, gr.update())


def generate_code_mlx_ui(
    jira_url: str,
    jira_token: str,
    test_key: str,
    platform: str = "ios",
    progress=gr.Progress()
):
    """파인튜닝 Qwen2.5-Coder LoRA로 Zephyr TC → Python 코드 생성"""
    if not jira_url or not jira_token or not test_key:
        yield "# ❌ 오류\n\nJira URL, PAT, 테스트 케이스 키를 모두 입력해주세요.", gr.update()
        return

    try:
        progress(0.1, desc="Jira에서 TC 가져오는 중...")
        yield f"# 🔄 코드 생성 (Qwen2.5-Coder LoRA)\n\n**Jira에서 `{test_key}` 가져오는 중...**", gr.update()

        client = JiraClient(base_url=jira_url.strip(), api_token=jira_token.strip())
        tc = client.get_test_case(test_key.strip(), debug=False)

        progress(0.3, desc="Qwen2.5-Coder LoRA 로드 중...")
        yield (
            "# 🔄 코드 생성 (Qwen2.5-Coder LoRA)\n\n"
            f"**✓ `{test_key}` 로드 완료** ({len(tc.steps)}개 스텝)\n\n"
            "**⏳ Qwen2.5-Coder-7B + LoRA로 코드 생성 중...**\n\n"
            "*모델 첫 실행 시 30-60초 로드 소요*"
        ), gr.update()

        # 학습=추론 일치: JiraTestCase를 collect_data.tc_to_training_format으로
        # 변환하고 플랫폼별 elements/app_settings 컨텍스트를 주입해 생성.
        from mlx_code_generator import MLXCodeGenerator
        generator = MLXCodeGenerator()
        code = generator.generate_from_jira_tc(tc, platform=platform)

        if not code:
            yield "# ❌ 코드 생성 실패\n\n생성된 코드가 비어있습니다.", gr.update()
            return

        progress(0.9, desc="결과 저장 중...")

        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tc_id = test_key.replace("-", "_").lower()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_test_{tc_id}_{ts}.py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            out_path = f.name

        display = (
            f"# ✅ 코드 생성 완료 (Qwen2.5-Coder LoRA)\n\n"
            f"**TC**: `{test_key}`  |  **플랫폼**: {platform}\n\n"
            "---\n\n"
            f"```python\n{code}\n```"
        )

        progress(1.0, desc="완료!")
        yield display, out_path

    except Exception as e:
        import traceback
        yield f"# ❌ 오류\n\n```\n{e}\n\n{traceback.format_exc()}\n```", gr.update()


def _load_tc_json(tc_json_file):
    """업로드용 TC JSON 로드 (생성 탭의 다운로드 파일 경로)."""
    if not tc_json_file:
        raise ValueError("먼저 TC를 생성하세요 (생성 결과 JSON이 없습니다).")
    path = tc_json_file.name if hasattr(tc_json_file, "name") else tc_json_file
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "raw" in data and "requirements" not in data:
        raise ValueError("생성 결과가 파싱 실패본(raw)입니다. 파싱 성공한 TC만 업로드 가능합니다.")
    return data


def test_zephyr_connection_ui(jira_url, jira_token):
    """Zephyr 업로드 전 Jira 연결 테스트."""
    if not jira_url or not jira_token:
        return "⚠️ Jira URL과 PAT를 입력하세요."
    try:
        from jira_client import JiraClient
        ok, msg = JiraClient(jira_url, jira_token).test_connection()
        return ("✅ " if ok else "❌ ") + msg
    except Exception as e:
        return f"❌ 연결 오류: {e}"


def upload_tc_to_zephyr_ui(jira_url, jira_token, mode, project_key, existing_key, tc_json_file, dry_run,
                           issue_type="Test case", severity="medium", fix_version="", component=""):
    """생성된 TC를 Zephyr Squad에 업로드 (dry_run이면 미리보기만)."""
    try:
        if not jira_url or not jira_token:
            return "# ⚠️ 입력 필요\n\nJira URL과 PAT 토큰을 입력하세요."
        data = _load_tc_json(tc_json_file)
        itype = (issue_type or "Test case").strip()
        fix_versions = [v.strip() for v in (fix_version or "").split(",") if v.strip()]
        components = [c.strip() for c in (component or "").split(",") if c.strip()]

        use_existing = (mode == "기존 키에 스텝 추가")
        pk = (project_key or "").strip()
        ek = (existing_key or "").strip()
        if use_existing and not ek:
            return "# ⚠️ 입력 필요\n\n기존 키 모드: 테스트 케이스 키(예: PROJ-123)를 입력하세요."
        if not use_existing and not pk:
            return "# ⚠️ 입력 필요\n\n새 이슈 모드: 프로젝트 키(예: QA)를 입력하세요."

        from jira_client import JiraClient
        client = JiraClient(jira_url, jira_token)
        # dry-run 미리보기는 네트워크 없이 동작. 실제 업로드 때만 연결 검증.
        msg = "(dry-run, 연결 검증 생략)"
        if not dry_run:
            ok, msg = client.test_connection()
            if not ok:
                return f"# ❌ 연결 실패\n\n{msg}"

        results = client.upload_generated_tc(
            data,
            project_key="" if use_existing else pk,
            existing_key=ek if use_existing else "",
            dry_run=dry_run,
            issue_type=itype,
            severity=(severity or "medium"),
            fix_versions=fix_versions,
            components=components,
        )

        head = "# 🔍 업로드 미리보기 (dry-run)\n\n실제 생성 안 됨. 확인 후 dry-run 해제하고 업로드하세요.\n\n" \
            if dry_run else "# ✅ Zephyr 업로드 완료\n\n"
        fv = ", ".join(fix_versions) if fix_versions else "-"
        comp = ", ".join(components) if components else "-"
        lines = [head, f"> 연결: {msg} | fixVersion: {fv} | Component: {comp}\n", f"> TC {len(results)}개\n\n",
                 "| # | 키 | 우선순위 | 레이블 | 스텝 | 제목 |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(results, 1):
            key = r.get("key") or r.get("target") or "(새 이슈)"
            lines.append(f"| {i} | {key} | {r.get('priority','-')} | {r.get('label','-')} | "
                         f"{r.get('steps_added',0)} | {r.get('summary','')[:36]} |")
        return "\n".join(lines)
    except Exception as e:
        import traceback
        return f"# ❌ 업로드 오류\n\n```\n{e}\n\n{traceback.format_exc()}\n```"


# ── 디자인: s12works 대시보드 톤 (Tailwind gray + 블랙 액센트, rounded-xl, 보더, 그림자X) ──
_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.gray,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_md,
    font=["Pretendard Variable", "Pretendard", "-apple-system", "system-ui",
          "Apple SD Gothic Neo", "Malgun Gothic", "sans-serif"],
).set(
    body_background_fill="#fafafa",
    body_text_color="#1a1a1a",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e5e7eb",
    block_shadow="none",
    block_label_text_weight="600",
    block_label_text_color="#374151",
    block_title_text_weight="600",
    input_background_fill="#ffffff",
    input_border_color="#e5e7eb",
    button_primary_background_fill="#111827",
    button_primary_background_fill_hover="#1f2937",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_border_color="#e5e7eb",
    button_secondary_text_color="#374151",
)

_CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; background:#fafafa; }
#hero { display:flex; align-items:center; gap:13px; padding:14px 4px 12px;
  border-bottom:1px solid #ececec; margin-bottom:8px; }
#hero .mark { width:40px; height:40px; border-radius:11px; background:#111827; color:#fff;
  display:flex; align-items:center; justify-content:center; font-size:21px; flex-shrink:0; }
#hero .title { font-size:20px; font-weight:700; color:#111827; letter-spacing:-.01em; line-height:1.2; }
#hero .sub { font-size:13px; color:#6b7280; margin-top:2px; }
.section-label { font-weight:600; font-size:12px; color:#6b7280;
  text-transform:uppercase; letter-spacing:.04em; margin:2px 0 -2px; }
.modelcard { background:#fff; border:1px solid #e5e7eb; border-radius:12px;
  padding:14px 16px; font-size:13px; line-height:1.5; }
.modelcard b { color:#111827; }
.tabitem { padding-top: 12px; }
button.selected { color:#111827 !important; font-weight:600 !important; }
.prose h1, .prose h2 { color:#111827; }
footer { display:none !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#d1d5db; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#9ca3af; }
"""


# Gradio 인터페이스 구성
def create_interface():
    with gr.Blocks(title="TC Studio") as demo:
        gr.HTML(
            "<div id='hero'>"
            "<div class='mark'>🧪</div>"
            "<div>"
            "<div class='title'>TC Studio</div>"
            "<div class='sub'>기획서 PDF → Zephyr 테스트케이스 · Jira TC → Python Appium 코드</div>"
            "</div></div>"
        )

        with gr.Tabs():
            # 탭 3: TC 생성
            with gr.Tab("📝 TC 생성", elem_classes="tabitem"):

                with gr.Row(equal_height=False):
                    # 좌측 입력부: 좁게
                    with gr.Column(scale=2, min_width=280):
                        gr.Markdown("입력", elem_classes="section-label")
                        gr.HTML(
                            "<div class='modelcard'>"
                            "<b>EXAONE 3.5 7.8B + LoRA</b> · 파인튜닝 모델<br>"
                            "<span style='color:#667085'>PDF 텍스트 추출(이미지는 VLM OCR) "
                            "→ 분석 → Zephyr TC</span></div>"
                        )
                        pdf_file_4 = gr.File(
                            label="서비스 기획서 PDF",
                            file_types=[".pdf"]
                        )
                        page_range_4 = gr.Textbox(
                            label="페이지 범위",
                            placeholder="예: 3 · 1-3 · 1-3,5  (비우면 전체)",
                            value=""
                        )
                        tc_gen_btn_4 = gr.Button("📝 TC 생성", variant="primary", size="lg")

                    # 우측 결과부: 넓게 + 여러 페이지면 페이지네이션
                    with gr.Column(scale=5):
                        gr.Markdown("결과", elem_classes="section-label")
                        tc_page_selector = gr.Radio(
                            choices=[], label="페이지", visible=False,
                            interactive=True
                        )
                        tc_pages_state = gr.State({})
                        tc_result_4 = gr.Markdown(
                            "← 왼쪽에서 PDF를 올리고 **TC 생성**을 눌러주세요.",
                            label="생성된 테스트 케이스"
                        )
                        tc_download_4 = gr.File(label="결과 다운로드", visible=False)

                tc_gen_btn_4.click(
                    fn=generate_tc_mlx_ui,
                    inputs=[pdf_file_4, page_range_4],
                    outputs=[tc_result_4, tc_download_4, tc_pages_state, tc_page_selector]
                )

                def _show_tc_page(label, pages):
                    if not pages or label not in pages:
                        return gr.update()
                    return pages[label]

                tc_page_selector.change(
                    fn=_show_tc_page,
                    inputs=[tc_page_selector, tc_pages_state],
                    outputs=[tc_result_4]
                )

                # ── Zephyr 업로드 (생성된 TC → Zephyr Squad) ──
                gr.Markdown("---")
                with gr.Accordion("📤 Zephyr 업로드 (생성된 TC를 Jira/Zephyr Squad에 올리기)", open=False):
                    gr.Markdown(
                        "위에서 **생성한 TC**(결과 다운로드 JSON)를 본인 PAT로 Zephyr Squad에 업로드합니다. "
                        "먼저 **연결 테스트**와 **미리보기(dry-run)**로 확인 후 업로드하세요."
                    )
                    with gr.Row():
                        zep_url = gr.Textbox(
                            label="Jira Base URL",
                            placeholder="https://jira.yourcompany.com",
                            value=os.getenv("JIRA_BASE_URL", ""), scale=2)
                        zep_pat = gr.Textbox(
                            label="Personal Access Token (PAT)", type="password",
                            placeholder="본인 Bearer 토큰",
                            value=os.getenv("JIRA_API_TOKEN", ""), scale=2)
                        zep_conn_btn = gr.Button("🔌 연결 테스트", scale=1)
                    zep_conn_out = gr.Markdown()
                    with gr.Row():
                        zep_mode = gr.Radio(
                            choices=["새 Test 이슈 생성", "기존 키에 스텝 추가"],
                            value="새 Test 이슈 생성", label="업로드 방식", scale=2)
                        zep_project = gr.Textbox(
                            label="프로젝트 키 (새 이슈)", placeholder="예: QA", scale=1)
                        zep_existing = gr.Textbox(
                            label="테스트 케이스 키 (기존)", placeholder="예: QA-123", scale=1)
                        zep_issuetype = gr.Textbox(
                            label="이슈 타입 (새 이슈)", value="Test case",
                            info="사내 Jira의 테스트 이슈타입명 (예: Test case / Test / 테스트)", scale=1)
                    with gr.Row():
                        zep_severity = gr.Dropdown(
                            choices=["high", "medium", "low"], value="medium",
                            label="심각도(우선순위)",
                            info="high→Major·중요도_상 / medium→Minor·중요도_중 / low→Trivial·중요도_하", scale=1)
                        zep_fixversion = gr.Textbox(
                            label="fixVersion (선택)", placeholder="예: 1.2.0 (쉼표로 여러 개)",
                            info="Jira fixVersion에 등록 (해당 버전이 프로젝트에 있어야 함)", scale=1)
                        zep_component = gr.Textbox(
                            label="Component (선택)", placeholder="예: 검색 (쉼표로 여러 개)",
                            info="Jira 컴포넌트에 등록 (해당 컴포넌트가 프로젝트에 있어야 함)", scale=1)
                    with gr.Row():
                        zep_dry = gr.Checkbox(label="미리보기(dry-run) — 실제 생성 안 함", value=True)
                        zep_upload_btn = gr.Button("📤 Zephyr 업로드", variant="primary")
                    zep_result = gr.Markdown()

                    zep_conn_btn.click(fn=test_zephyr_connection_ui,
                                       inputs=[zep_url, zep_pat], outputs=[zep_conn_out])
                    zep_upload_btn.click(
                        fn=upload_tc_to_zephyr_ui,
                        inputs=[zep_url, zep_pat, zep_mode, zep_project, zep_existing,
                                tc_download_4, zep_dry, zep_issuetype, zep_severity,
                                zep_fixversion, zep_component],
                        outputs=[zep_result])

            # 탭 5: TC Generator - Python 테스트 코드 생성
            with gr.Tab("🔧 테스트 코드 생성", elem_classes="tabitem"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=280):
                        gr.Markdown("입력", elem_classes="section-label")
                        gr.HTML(
                            "<div class='modelcard'>"
                            "<b>Qwen2.5-Coder-7B + LoRA</b> · 파인튜닝 모델<br>"
                            "<span style='color:#667085'>Jira TC 조회 → step_reporter·"
                            "WebDriverWait·로케이터 패턴 → Python Appium 코드</span></div>"
                        )
                        jira_url_6 = gr.Textbox(
                            label="Jira Base URL",
                            placeholder="https://jira.yourcompany.com",
                            value=os.getenv("JIRA_BASE_URL", "")
                        )
                        jira_token_6 = gr.Textbox(
                            label="Personal Access Token (PAT)",
                            type="password",
                            placeholder="Bearer 토큰",
                            value=os.getenv("JIRA_API_TOKEN", "")
                        )
                        test_key_6 = gr.Textbox(
                            label="테스트 케이스 키",
                            placeholder="QA-12345"
                        )
                        platform_6 = gr.Radio(
                            choices=["ios", "android"],
                            value="ios",
                            label="대상 플랫폼",
                        )
                        generate_btn_6 = gr.Button("🚀 Python 코드 생성", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        gr.Markdown("결과", elem_classes="section-label")
                        result_6 = gr.Markdown(
                            "← 테스트 케이스 키를 입력하고 **코드 생성**을 눌러주세요.",
                            label="생성된 코드"
                        )
                        download_6 = gr.File(label="코드 다운로드", visible=False)

                generate_btn_6.click(
                    fn=generate_code_mlx_ui,
                    inputs=[
                        jira_url_6,
                        jira_token_6,
                        test_key_6,
                        platform_6,
                    ],
                    outputs=[result_6, download_6]
                )

                gr.Markdown(
                    "<span style='color:#667085;font-size:13px'>"
                    "학습 데이터 20개 iOS TC-코드 쌍 · 첫 실행 시 모델 로드 30-60초 · "
                    "플랫폼별 elements.py·app_settings.py 자동 참조</span>"
                )

        with gr.Accordion("📚 도움말 · 페이지 범위 / 평가 기준", open=False):
            gr.Markdown(
                "**페이지 범위 형식**\n"
                "- `1-3` → 1·2·3페이지 / `5` → 5페이지만 / `1-3,5,7-10` → 1·2·3·5·7·8·9·10\n\n"
                "**TC 평가 기준**\n"
                "1. 커버리지 — 기획서의 모든 기능이 포함되었는가\n"
                "2. 절차 — 테스트 단계가 논리적이고 실행 가능한가\n"
                "3. 사전조건 — 테스트 환경·데이터가 명확한가\n"
                "4. 명확성 — 처음 보는 사람도 이해할 수 있는가\n"
                "5. 표준양식 — 조직의 표준 양식을 준수하는가"
            )

    return demo


if __name__ == "__main__":
    # Gradio 앱 실행
    demo = create_interface()
    demo.queue()  # 비동기 작업을 위한 큐 활성화

    # 포트 설정 (환경변수 또는 기본값 7860 고정)
    port = int(os.environ.get("GRADIO_PORT", "7860"))

    # 네트워크 IP 주소 확인
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"

    # Share 모드 설정 (환경변수 또는 기본값)
    use_share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    print(f"\n🚀 GUI 서버 시작 중... 포트: {port}")
    print(f"🌐 로컬 접속: http://localhost:{port}")
    print(f"🌐 네트워크 접속: http://{local_ip}:{port}")

    if use_share:
        print(f"\n⚠️  Share 모드 활성화됨!")
        print(f"⚠️  공개 URL이 생성됩니다. 인증 설정을 확인하세요!")
    else:
        print(f"📌 팀원들과 공유할 URL: http://{local_ip}:{port}")
        print(f"\n💡 내부망에서 IP 접속이 안 되면: GRADIO_SHARE=true python3 gui.py")

    print()

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=use_share,  # 환경변수로 제어
        show_error=True,
        inbrowser=True,  # 브라우저 자동 열기
        theme=_THEME,     # Gradio 6: 테마/CSS는 launch에서 적용
        css=_CSS,
        auth=("admin", "test1234") if use_share else None,  # Share 모드에서는 인증 필수
    )
