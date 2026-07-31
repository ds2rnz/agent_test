"""PDF 파일을 DOCX와 TXT 바이트 데이터로 변환하는 도우미."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Mm, Pt
from pypdf import PdfReader


MAX_PDF_SIZE = 50 * 1024 * 1024


class PdfConversionError(ValueError):
    """사용자에게 안내할 수 있는 PDF 변환 오류."""


@dataclass(frozen=True)
class PdfConversionResult:
    """한 PDF의 변환 결과."""

    source_name: str
    docx_name: str
    txt_name: str
    docx_bytes: bytes
    txt_bytes: bytes
    page_count: int
    text_page_count: int


def _safe_stem(file_name: str) -> str:
    """다운로드 파일명에 사용할 안전한 이름을 만듭니다."""
    stem = Path(file_name).stem.strip() or "converted_pdf"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    return stem[:120]


def _normalize_text(text: str) -> str:
    """PDF 추출 과정에서 생기는 과도한 공백을 정리합니다."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pdf_pages(pdf_bytes: bytes) -> list[str]:
    """PDF의 각 페이지에서 텍스트를 추출합니다."""
    if not pdf_bytes:
        raise PdfConversionError("PDF 파일이 비어 있습니다.")
    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise PdfConversionError("PDF 파일은 50MB 이하만 변환할 수 있습니다.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as error:
        raise PdfConversionError(
            "PDF를 열 수 없습니다. 손상되었거나 암호화된 파일인지 확인해 주세요."
        ) from error

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as error:
            raise PdfConversionError(
                "암호화된 PDF는 변환할 수 없습니다. 암호를 해제해 주세요."
            ) from error
        if not unlocked:
            raise PdfConversionError(
                "암호화된 PDF는 변환할 수 없습니다. 암호를 해제해 주세요."
            )

    if not reader.pages:
        raise PdfConversionError("PDF에 페이지가 없습니다.")

    pages = []
    for page in reader.pages:
        try:
            pages.append(_normalize_text(page.extract_text() or ""))
        except Exception:
            pages.append("")

    if not any(pages):
        raise PdfConversionError(
            "PDF에서 텍스트를 찾지 못했습니다. "
            "스캔 이미지 PDF라면 OCR 처리가 먼저 필요합니다."
        )
    return pages


def _build_txt(pages: list[str]) -> bytes:
    sections = []
    for page_number, text in enumerate(pages, start=1):
        content = text or "[이 페이지에서 추출된 텍스트가 없습니다.]"
        sections.append(f"===== {page_number}페이지 =====\n{content}")
    # utf-8-sig는 Windows 메모장에서 한글이 깨지는 문제를 줄여 줍니다.
    return ("\n\n".join(sections) + "\n").encode("utf-8-sig")


def _build_docx(source_name: str, pages: list[str]) -> bytes:
    document = Document()
    section = document.sections[0]
    section.top_margin = Mm(20)
    section.bottom_margin = Mm(20)
    section.left_margin = Mm(22)
    section.right_margin = Mm(22)

    for style_name in ("Normal", "Title", "Heading 1"):
        style = document.styles[style_name]
        style.font.name = "맑은 고딕"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")

    normal_style = document.styles["Normal"]
    normal_style.font.size = Pt(10.5)
    normal_style.paragraph_format.space_after = Pt(6)
    normal_style.paragraph_format.line_spacing = 1.15

    title = document.add_paragraph()
    title.style = document.styles["Title"]
    title.add_run(Path(source_name).name)
    document.add_paragraph(f"PDF 변환 문서 · 총 {len(pages)}페이지")

    for page_number, text in enumerate(pages, start=1):
        if page_number > 1:
            document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

        document.add_heading(f"{page_number}페이지", level=1)
        if not text:
            paragraph = document.add_paragraph(
                "[이 페이지에서 추출된 텍스트가 없습니다.]"
            )
            paragraph.runs[0].italic = True
            continue

        for block in text.split("\n"):
            document.add_paragraph(block if block else " ")

    output = BytesIO()
    document.save(output)
    return output.getvalue()


def convert_pdf(pdf_bytes: bytes, file_name: str) -> PdfConversionResult:
    """PDF 하나를 DOCX와 TXT로 변환합니다."""
    if Path(file_name).suffix.lower() != ".pdf":
        raise PdfConversionError("PDF 파일만 변환할 수 있습니다.")

    pages = extract_pdf_pages(pdf_bytes)
    stem = _safe_stem(file_name)
    return PdfConversionResult(
        source_name=file_name,
        docx_name=f"{stem}.docx",
        txt_name=f"{stem}.txt",
        docx_bytes=_build_docx(file_name, pages),
        txt_bytes=_build_txt(pages),
        page_count=len(pages),
        text_page_count=sum(bool(page) for page in pages),
    )
