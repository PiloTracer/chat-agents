# app/routers/documents.py
from __future__ import annotations
from typing import List, Tuple, Dict
import os, shutil, tempfile, re, io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.auth import Principal
from app.db import get_db
from app.parsing import parse_file
from app.rag import upsert_document
from app.security import ensure_agent_access, require_role, Role
from app.models import Document, Chunk
import fitz  # PyMuPDF
from fastapi.responses import Response
import io
import fitz  # PyMuPDF

router = APIRouter()

def _norm_content_type(filename: str, content_type: str | None) -> str:
    ct = (content_type or "").lower().strip()
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return "application/pdf"
    if name.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if name.endswith(".pptx"):
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if name.endswith(".rtf"):
        return "application/rtf"
    if name.endswith(".htm") or name.endswith(".html"):
        return "text/html"
    return ct or "application/octet-stream"


@router.post("/upload")
async def upload(
    agent_slug: str = Form(...),
    files: List[UploadFile] = File(...),
    principal: Principal = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    ensure_agent_access(db, principal, agent_slug)

    if not files:
        raise HTTPException(status_code=422, detail="No files provided")

    results: List[dict[str, object]] = []
    total_chunks = 0

    for upload_file in files:
        tmp_path: str | None = None
        n_chunks = 0
        doc_id = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                shutil.copyfileobj(upload_file.file, tmp)
                tmp_path = tmp.name

            ct = _norm_content_type(upload_file.filename or "", upload_file.content_type)

            parsed_iter = parse_file(tmp_path, ct)
            parsed: List[Tuple[str, str]] = [
                (src, txt) for (src, txt) in parsed_iter if txt and txt.strip()
            ]
            try:
                doc_id, n_chunks = await upsert_document(
                    db,
                    agent_slug,
                    upload_file.filename,
                    ct,
                    parsed,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        if n_chunks == 0:
            raise HTTPException(422, f"No extractable text from {upload_file.filename}")

        results.append(
            {
                "filename": upload_file.filename,
                "document_id": doc_id,
                "chunks": n_chunks,
            }
        )
        total_chunks += n_chunks

    return {"ok": True, "documents": results, "total_chunks": total_chunks}


_PAGE_RE = re.compile(r"(?:^|#)(page|slide|para|sec)=(\d+)")


def _extract_page(source_ref: str) -> int | None:
    if not source_ref:
        return None
    m = _PAGE_RE.search(source_ref)
    if not m:
        return None
    try:
        return int(m.group(2))
    except Exception:
        return None


@router.get("/{document_id}/map")
def document_map(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.ord.asc())
        .all()
    )
    pages: Dict[int, List[Dict[str, object]]] = {}
    no_page: List[Dict[str, object]] = []
    for r in rows:
        p = _extract_page(getattr(r, "source_ref", "") or "")
        entry = {
            "chunk_id": r.id,
            "ord": r.ord or 0,
            "source_ref": getattr(r, "source_ref", "") or "",
            "length": len(getattr(r, "text", "") or ""),
        }
        if p is not None:
            pages.setdefault(p, []).append(entry)
        else:
            no_page.append(entry)
    page_list = [
        {"page": page_no, "chunks": sorted(items, key=lambda x: int(x["ord"]))}
        for page_no, items in sorted(pages.items(), key=lambda kv: kv[0])
    ]
    if no_page:
        page_list.append({"page": None, "chunks": sorted(no_page, key=lambda x: int(x["ord"]))})
    # Detect missing pages (gaps)
    page_numbers = sorted([p for p in pages.keys() if isinstance(p, int)])
    missing_pages: List[int] = []
    if page_numbers:
        first = page_numbers[0]
        last = page_numbers[-1]
        have = set(page_numbers)
        missing_pages = [n for n in range(first, last + 1) if n not in have]
    return {
        "ok": True,
        "document": {"id": doc.id, "filename": doc.filename, "agent_slug": doc.agent_slug},
        "pages": page_list,
        "totals": {"chunks": len(rows), "pages": len(page_list)},
        "missing_pages": missing_pages,
        "continuity_ok": len(missing_pages) == 0,
    }


@router.get("/{document_id}/consolidated")
def document_consolidated(
    document_id: int,
    start: int | None = None,
    end: int | None = None,
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.ord.asc())
        .all()
    )
    # Group by page
    pages: Dict[int, List[str]] = {}
    no_page: List[str] = []
    for r in rows:
        p = _extract_page(getattr(r, "source_ref", "") or "")
        if p is not None:
            pages.setdefault(p, []).append(getattr(r, "text", "") or "")
        else:
            no_page.append(getattr(r, "text", "") or "")
    lines: List[str] = []
    for page_no, texts in sorted(pages.items(), key=lambda kv: kv[0]):
        if start is not None and page_no < start:
            continue
        if end is not None and page_no > end:
            continue
        lines.append(f"=== Page {page_no} ===")
        lines.append("\n\n".join(texts))
        lines.append("")
    if no_page:
        lines.append("=== Unpaged ===")
        lines.append("\n\n".join(no_page))
        lines.append("")
    return {
        "ok": True,
        "document": {"id": doc.id, "filename": doc.filename, "agent_slug": doc.agent_slug},
        "text": "\n".join(lines).strip(),
    }


def _build_consolidated_text(db: Session, document_id: int, start: int | None = None, end: int | None = None) -> Tuple[Document, str]:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.ord.asc())
        .all()
    )
    pages: Dict[int, List[str]] = {}
    no_page: List[str] = []
    for r in rows:
        p = _extract_page(getattr(r, "source_ref", "") or "")
        if p is not None:
            pages.setdefault(p, []).append(getattr(r, "text", "") or "")
        else:
            no_page.append(getattr(r, "text", "") or "")
    lines: List[str] = []
    for page_no, texts in sorted(pages.items(), key=lambda kv: kv[0]):
        if start is not None and page_no < start:
            continue
        if end is not None and page_no > end:
            continue
        lines.append(f"=== Page {page_no} ===")
        lines.append("\n\n".join(texts))
        lines.append("")
    if no_page and start is None and end is None:
        lines.append("=== Unpaged ===")
        lines.append("\n\n".join(no_page))
        lines.append("")
    return doc, "\n".join(lines).strip()


@router.get("/{document_id}/pages")
def document_pages(
    document_id: int,
    start: int | None = None,
    end: int | None = None,
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.ord.asc())
        .all()
    )
    output: List[Dict[str, object]] = []
    for r in rows:
        p = _extract_page(getattr(r, "source_ref", "") or "")
        if p is not None:
            if start is not None and p < start:
                continue
            if end is not None and p > end:
                continue
        output.append({
            "chunk_id": r.id,
            "document_id": r.document_id,
            "page_number": p,
            "ord": r.ord or 0,
            "source_file": doc.filename,
            "source_ref": getattr(r, "source_ref", "") or "",
            "text": getattr(r, "text", "") or "",
        })
    return {"ok": True, "document": {"id": doc.id, "filename": doc.filename, "agent_slug": doc.agent_slug}, "chunks": output}


@router.get("/{document_id}/verify")
def document_verify(
    document_id: int,
    start: int | None = None,
    end: int | None = None,
    db: Session = Depends(get_db),
):
    mapping = document_map(document_id, db)
    pages = [p.get("page") for p in mapping.get("pages", []) if isinstance(p.get("page"), int)]
    if not pages:
        return {"ok": True, "continuity_ok": True, "missing_pages": []}
    first = min(pages) if start is None else start
    last = max(pages) if end is None else end
    have = set(p for p in pages if (start is None or p >= start) and (end is None or p <= end))
    missing = [n for n in range(first, last + 1) if n not in have]
    return {"ok": True, "from": first, "to": last, "missing_pages": missing, "continuity_ok": len(missing) == 0}


@router.get("/{document_id}/export.{fmt}")
def document_export(
    document_id: int,
    fmt: str,
    start: int | None = None,
    end: int | None = None,
    db: Session = Depends(get_db),
):
    doc, text = _build_consolidated_text(db, document_id, start, end)

    filename_base = (doc.filename or f"document-{doc.id}").rsplit("/", 1)[-1]
    if fmt.lower() in {"txt", "text"}:
        data = text.encode("utf-8")
        headers = {"Content-Disposition": f"attachment; filename=\"{filename_base}.txt\""}
        return Response(content=data, media_type="text/plain; charset=utf-8", headers=headers)

    if fmt.lower() in {"md", "markdown"}:
        md_lines = [f"# {filename_base}", "", text]
        data = "\n".join(md_lines).encode("utf-8")
        headers = {"Content-Disposition": f"attachment; filename=\"{filename_base}.md\""}
        return Response(content=data, media_type="text/markdown; charset=utf-8", headers=headers)

    if fmt.lower() == "pdf":
        # Render a simple PDF using PyMuPDF
        page_w, page_h = 595, 842  # A4 in points
        margin = 36
        line_h = 14
        max_lines = max(1, int((page_h - 2 * margin) / line_h))
        lines = text.split("\n")

        doc_pdf = fitz.open()
        cursor = 0
        while cursor < len(lines):
            page = doc_pdf.new_page(width=page_w, height=page_h)
            start_idx = cursor
            end_idx = min(len(lines), start_idx + max_lines)
            chunk = "\n".join(lines[start_idx:end_idx])
            rect = fitz.Rect(margin, margin, page_w - margin, page_h - margin)
            page.insert_textbox(rect, chunk, fontsize=10, fontname="helv", align=0)
            cursor = end_idx

        buf = io.BytesIO()
        doc_pdf.save(buf)
        doc_pdf.close()
        data = buf.getvalue()
        headers = {"Content-Disposition": f"attachment; filename=\"{filename_base}.pdf\""}
        return Response(content=data, media_type="application/pdf", headers=headers)

    raise HTTPException(status_code=400, detail="Unsupported format. Use txt, md or pdf.")


def _build_consolidated_text(db: Session, document_id: int) -> Tuple[Document, str]:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.ord.asc())
        .all()
    )
    pages: Dict[int, List[str]] = {}
    no_page: List[str] = []
    for r in rows:
        p = _extract_page(getattr(r, "source_ref", "") or "")
        if p is not None:
            pages.setdefault(p, []).append(getattr(r, "text", "") or "")
        else:
            no_page.append(getattr(r, "text", "") or "")
    lines: List[str] = []
    for page_no, texts in sorted(pages.items(), key=lambda kv: kv[0]):
        lines.append(f"=== Page {page_no} ===")
        lines.append("\n\n".join(texts))
        lines.append("")
    if no_page:
        lines.append("=== Unpaged ===")
        lines.append("\n\n".join(no_page))
        lines.append("")
    return doc, "\n".join(lines).strip()


@router.get("/{document_id}/export.{fmt}")
def document_export(document_id: int, fmt: str, db: Session = Depends(get_db)):
    doc, text = _build_consolidated_text(db, document_id)

    filename_base = (doc.filename or f"document-{doc.id}").rsplit("/", 1)[-1]
    if fmt.lower() in {"txt", "text"}:
        data = text.encode("utf-8")
        headers = {"Content-Disposition": f"attachment; filename=\"{filename_base}.txt\""}
        return Response(content=data, media_type="text/plain; charset=utf-8", headers=headers)

    if fmt.lower() in {"md", "markdown"}:
        md_lines = [f"# {filename_base}", "", text]
        data = "\n".join(md_lines).encode("utf-8")
        headers = {"Content-Disposition": f"attachment; filename=\"{filename_base}.md\""}
        return Response(content=data, media_type="text/markdown; charset=utf-8", headers=headers)

    if fmt.lower() == "pdf":
        # Render a simple PDF using PyMuPDF
        # Page: A4 portrait (595 x 842 pt)
        page_w, page_h = 595, 842
        margin = 36
        line_h = 14
        max_lines = max(1, int((page_h - 2 * margin) / line_h))
        lines = text.split("\n")

        doc_pdf = fitz.open()
        cursor = 0
        while cursor < len(lines):
            page = doc_pdf.new_page(width=page_w, height=page_h)
            start = cursor
            end = min(len(lines), start + max_lines)
            chunk = "\n".join(lines[start:end])
            rect = fitz.Rect(margin, margin, page_w - margin, page_h - margin)
            page.insert_textbox(rect, chunk, fontsize=10, fontname="helv", align=0)
            cursor = end

        buf = io.BytesIO()
        doc_pdf.save(buf)
        doc_pdf.close()
        data = buf.getvalue()
        headers = {"Content-Disposition": f"attachment; filename=\"{filename_base}.pdf\""}
        return Response(content=data, media_type="application/pdf", headers=headers)

    raise HTTPException(status_code=400, detail="Unsupported format. Use txt, md or pdf.")
