# app/parsing.py
from __future__ import annotations
from typing import Iterable, Iterator, Tuple, List
import os, io, re, zipfile
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
from docx import Document as DocxDocument
from pptx import Presentation
import fitz  # PyMuPDF
from PIL import Image


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _iter_nonempty(pairs: Iterable[Tuple[str, str]]) -> Iterator[Tuple[str, str]]:
    for src, txt in pairs:
        t = _clean_text(txt)
        if t:
            yield (src, t)


# ---------- HTML ----------

def iter_html(path: str) -> Iterator[Tuple[str, str]]:
    with open(path, "rb") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for bad in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        bad.decompose()
    text = soup.get_text("\n")
    yield (os.path.basename(path), text)


# ---------- RTF ----------

def _iter_rtf_images(raw: bytes, base_name: str) -> Iterator[Tuple[str, str]]:
    text = raw.decode("latin-1", errors="ignore")
    pict_re = re.compile(r"\\pict[^{]*?([0-9A-Fa-f\\s]+)}", re.DOTALL)
    for idx, match in enumerate(pict_re.finditer(text), start=1):
        hex_blob = re.sub(r"[^0-9A-Fa-f]", "", match.group(1))
        if len(hex_blob) < 20:
            continue
        try:
            data = bytes.fromhex(hex_blob)
        except ValueError:
            continue
        ocr = _ocr_image_bytes(data)
        if ocr:
            yield (f"{base_name}#image={idx}", ocr)


def iter_rtf(path: str) -> Iterator[Tuple[str, str]]:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        txt = rtf_to_text(raw.decode("utf-8", errors="ignore"))
    except Exception:
        txt = rtf_to_text(raw.decode("latin-1", errors="ignore"))
    base = os.path.basename(path)
    yield (base, txt)
    yield from _iter_rtf_images(raw, base)


# ---------- DOCX ----------

def _iter_docx_images(path: str) -> Iterator[Tuple[str, str]]:
    base = os.path.basename(path)
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if n.startswith("word/media/"))
            for idx, name in enumerate(names, start=1):
                try:
                    data = zf.read(name)
                except KeyError:
                    continue
                ocr = _ocr_image_bytes(data)
                if ocr:
                    yield (f"{base}#image={idx}", ocr)
    except zipfile.BadZipFile:
        return


def iter_docx(path: str) -> Iterator[Tuple[str, str]]:
    doc = DocxDocument(path)
    parts: List[str] = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text)
    for tbl in doc.tables:
        for row in tbl.rows:
            cells = [c.text.strip() for c in row.cells]
            line = " | ".join([c for c in cells if c])
            if line:
                parts.append(line)
    base = os.path.basename(path)
    if parts:
        yield (base, "\n".join(parts))
    yield from _iter_docx_images(path)


# ---------- PPTX ----------

def _iter_pptx_images(path: str) -> Iterator[Tuple[str, str]]:
    base = os.path.basename(path)
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if n.startswith("ppt/media/"))
            for idx, name in enumerate(names, start=1):
                try:
                    data = zf.read(name)
                except KeyError:
                    continue
                ocr = _ocr_image_bytes(data)
                if ocr:
                    yield (f"{base}#image={idx}", ocr)
    except zipfile.BadZipFile:
        return


def iter_pptx(path: str) -> Iterator[Tuple[str, str]]:
    prs = Presentation(path)
    for i, slide in enumerate(prs.slides, start=1):
        lines = []
        for shape in slide.shapes:
            if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    s = "".join(run.text for run in para.runs).strip()
                    if s:
                        lines.append(s)
        if lines:
            yield (f"{os.path.basename(path)}#slide={i}", "\n".join(lines))
    yield from _iter_pptx_images(path)


# ---------- PDF (text-first; OCR fallback) ----------

def _try_ocr(img: Image.Image) -> str:
    try:
        import pytesseract
    except Exception:
        return ""
    try:
        langs = os.getenv("TESS_LANGS", "eng")
        return pytesseract.image_to_string(img, lang=langs)
    except Exception:
        try:
            return pytesseract.image_to_string(img)
        except Exception:
            return ""


def _ocr_image_bytes(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as raw:
            img = raw.convert("L")
            try:
                return _try_ocr(img)
            finally:
                img.close()
    except Exception:
        return ""


def iter_pdf(path: str) -> Iterator[Tuple[str, str]]:
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        txt = page.get_text("text") or ""
        if txt.strip():
            yield (f"{os.path.basename(path)}#page={i}", txt)
        else:
            pix = page.get_pixmap(dpi=300)
            with Image.open(io.BytesIO(pix.tobytes("png"))) as raw:
                img = raw.convert("L")
                try:
                    ocr = _try_ocr(img)
                finally:
                    img.close()
            if ocr.strip():
                yield (f"{os.path.basename(path)}#page={i}", ocr)


# ---------- Router ----------

def parse_file(path: str, content_type: str) -> Iterator[Tuple[str, str]]:
    ct = (content_type or "").lower()
    name = os.path.basename(path).lower()
    if name.endswith((".html", ".htm")) or "html" in ct:
        yield from _iter_nonempty(iter_html(path))
    elif name.endswith(".rtf") or "rtf" in ct:
        yield from _iter_nonempty(iter_rtf(path))
    elif name.endswith(".docx") or "wordprocessingml.document" in ct:
        yield from _iter_nonempty(iter_docx(path))
    elif name.endswith(".pptx") or "presentationml.presentation" in ct:
        yield from _iter_nonempty(iter_pptx(path))
    elif name.endswith(".pdf") or "pdf" in ct:
        yield from _iter_nonempty(iter_pdf(path))
    else:
        with open(path, "rb") as f:
            data = f.read().decode("utf-8", errors="ignore")
        yield from _iter_nonempty([(os.path.basename(path), data)])
