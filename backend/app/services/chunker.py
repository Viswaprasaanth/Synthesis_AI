from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_paper(
    text: str,
    paper_id: str,
    paper_title: str,
    year: int | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 300,
) -> list[dict]:
    """
    Split paper text into chunks, each tagged with paper metadata.
    This is the key difference from single-doc RAG — every chunk
    knows which paper it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    raw_chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "metadata": {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "year": year,
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(raw_chunks)
    ]