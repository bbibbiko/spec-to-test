#!/usr/bin/env python3
"""OCR 캐시 하이브리드 재생성 — VLM OCR 캐시 → 텍스트 PDF는 pymupdf로 교체.

배경: VLM OCR이 텍스트 PDF의 표 행을 통째로 누락하는 경우가 있었다(→ 모델이 케이스를 부실하게 생성).
추론 경로는 이미 ocr_pdf_pages 하이브리드로 수정했고, 학습도 train=infer 를 맞추려면
학습 입력(OCR 캐시)도 동일한 하이브리드로 재생성해야 해서 이 스크립트를 사용한다.

동작: data_vlm_ocr/ocr_cache.json 의 각 (pdf::page) 항목에 대해
  - 소스 PDF가 있고 pymupdf 임베디드 텍스트 >= threshold → pymupdf 텍스트로 교체(완전·정확)
  - 아니면(스캔/이미지·PDF 없음) 기존 VLM 캐시 유지
→ data_vlm_ocr/ocr_cache_hybrid.json 저장 + 변경 리포트 출력.

주의: pymupdf로 내용이 크게 늘어난 페이지는 그 타깃이 예전(누락된) OCR 기준이라 sparse할 수 있다.
리포트의 'GAINED' 목록이 타깃 재검토 후보이므로 학습 전 확인을 권장한다.

사용법:
  python3 rebuild_ocr_cache_hybrid.py            # ocr_cache_hybrid.json 생성 + 리포트
  python3 rebuild_ocr_cache_hybrid.py --apply    # 추가로 ocr_cache.json 백업 후 교체
"""
import json
import re
import sys
import shutil
from pathlib import Path

FT = Path(__file__).parent
SRC = FT / "data_vlm_ocr/ocr_cache.json"
DST = FT / "data_vlm_ocr/ocr_cache_hybrid.json"
REPORT = FT / "data_vlm_ocr/ocr_cache_hybrid_report.json"
THRESHOLD = 150  # ocr_pdf_pages(infer)와 동일 기준


def looks_clean(text: str) -> bool:
    """모지베이크(폰트 깨짐) 판정 — infer_pipeline._looks_clean과 동일 기준."""
    if not text:
        return False
    good = len(re.findall(r'[가-힣A-Za-z0-9\s.,:/()\[\]·\-→%]', text))
    return good / len(text) >= 0.85


def name_to_path() -> dict:
    """raw_data/pair_*.json 의 spec_pdf.path 들 → {파일명: 경로} 맵."""
    m = {}
    for f in (FT / "raw_data").glob("pair_*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        sp = d.get("spec_pdf") or {}
        p = sp.get("path")
        if p:
            m[Path(p).name] = p
    return m


def main():
    import fitz
    cache = json.loads(SRC.read_text(encoding="utf-8"))
    n2p = name_to_path()

    new_cache = {}
    gained, replaced, kept, missing = [], 0, 0, 0
    for key, old in cache.items():
        if "::" not in key:
            new_cache[key] = old
            continue
        name, page = key.rsplit("::", 1)
        path = n2p.get(name)
        if not (path and Path(path).exists() and page.isdigit()):
            new_cache[key] = old
            missing += 1
            continue
        try:
            doc = fitz.open(path)
            pno = int(page)
            txt = doc[pno - 1].get_text().strip() if 1 <= pno <= len(doc) else ""
            doc.close()
        except Exception:
            new_cache[key] = old
            kept += 1
            continue
        if len(txt) >= THRESHOLD and looks_clean(txt):
            new_cache[key] = f"[p.{pno}]\n{txt}"
            replaced += 1
            old_len = len(old)
            if len(txt) > old_len * 1.25:  # 콘텐츠 크게 증가 → 타깃 재검증 후보
                gained.append({"key": key, "old_chars": old_len, "new_chars": len(txt)})
        else:
            new_cache[key] = old  # 이미지/스캔/폰트깨짐 페이지는 VLM 유지
            kept += 1

    DST.write_text(json.dumps(new_cache, ensure_ascii=False, indent=2), encoding="utf-8")
    gained.sort(key=lambda g: g["new_chars"] - g["old_chars"], reverse=True)
    REPORT.write_text(json.dumps({
        "threshold": THRESHOLD, "total": len(cache),
        "replaced_pymupdf": replaced, "kept_vlm": kept, "pdf_missing": missing,
        "gained_for_target_review": gained,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"총 {len(cache)} | pymupdf 교체 {replaced} | VLM 유지 {kept} | PDF없음 {missing}")
    print(f"콘텐츠 크게 증가(타깃 재검증 후보) {len(gained)}개:")
    for g in gained[:25]:
        print(f"  {g['key']}  {g['old_chars']}→{g['new_chars']}자")
    print(f"\n저장: {DST.name} / {REPORT.name}")

    if "--apply" in sys.argv:
        bak = SRC.with_suffix(".json.vlm_bak")
        shutil.copy2(SRC, bak)
        shutil.copy2(DST, SRC)
        print(f"✅ --apply: {SRC.name} 교체 (백업 {bak.name})")
    else:
        print("ℹ️  ocr_cache.json 교체하려면: python3 rebuild_ocr_cache_hybrid.py --apply")


if __name__ == "__main__":
    main()
