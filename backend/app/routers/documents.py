# app/routers/documents.py
from __future__ import annotations
from typing import List, Tuple
import os, shutil, tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from app.auth import Principal
from app.db import get_db
from app.parsing import parse_file
from app.rag import upsert_document
from app.security import ensure_agent_access, require_role, Role

router = APIRouter()


def _norm_content_type(filename: str, content_type: str | None) -> str:
    ct = (content_type or "").lower().strip()
    name = filename.lower()
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
    file: UploadFile = File(...),
    principal: Principal = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    ensure_agent_access(db, principal, agent_slug)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    ct = _norm_content_type(file.filename, file.content_type)

    try:
        parsed_iter = parse_file(tmp_path, ct)
        parsed: List[Tuple[str, str]] = [
            (src, txt) for (src, txt) in parsed_iter if txt and txt.strip()
        ]
        try:
            doc_id, n_chunks = await upsert_document(db, agent_slug, file.filename, ct, parsed)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if n_chunks == 0:
        raise HTTPException(422, f"No extractable text from {file.filename}")

    return {"ok": True, "document_id": doc_id, "chunks": n_chunks}
