from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.middleware.auth import require_api_key
from app.services.pdf_reader import extract_text_from_pdf
from app.services.paper_parser import parse_paper
from app.services.chunker import chunk_paper
from app.services.vector_store import upsert_chunks
from app.models import IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/", response_model=IngestResponse)
async def ingest_paper(
    file: UploadFile = File(...),
    _: str = Depends(require_api_key),
):
    """Upload a PDF → extract text → parse metadata → chunk → embed → store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # 1. Read PDF
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)

    if len(text.strip()) < 100:
        raise HTTPException(400, "Could not extract text from PDF.")

    # 2. Parse metadata via LLM
    paper_meta = await parse_paper(text)

    # 3. Chunk with paper metadata
    chunks = chunk_paper(
        text=text,
        paper_id=paper_meta.paper_id,
        paper_title=paper_meta.title,
        year=paper_meta.year,
    )

    # 4. Embed and store in Qdrant
    upsert_chunks(chunks)

    return IngestResponse(
        paper_id=paper_meta.paper_id,
        title=paper_meta.title,
        filename=file.filename,
        chunks_indexed=len(chunks),
    )