import json
import re
import unicodedata
from pathlib import Path
from pypdf import PdfReader

SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}
SUPPORTED_PDF_SUFFIXES = {".pdf"}

def project_paths(root=None):
    """Every path the project uses, in one dict.

    Relative to the notebooks/ folder by default, so the notebooks and the
    modules agree on where the corpus and the store live.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    documents = root / "data" / "documents"
    storage = root / "storage"

    documents.mkdir(parents=True, exist_ok=True)
    storage.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "documents": documents,
        "storage": storage,
        "chunks": storage / "chunks.json",
        "vectors": storage / "vectors.npy",
        "config": storage / "store_config.json",
        "faiss": storage / "index.faiss",
    }

def load_txt(path):
    """Load a .txt/.md file as a single Document."""
    path = Path(path)
    # errors="replace" instead of crashing: real corpora contain broken bytes,
    # and losing one character is better than losing the whole file.
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{
        "text": text,
        "metadata": {"source": path.name, "path": str(path), "page": None, "type": "text"},
    }]


def load_pdf(path):
    """Load a .pdf as one Document per page (pages are a natural unit)."""
    from pypdf import PdfReader  # imported lazily so .txt-only users need no pypdf

    path = Path(path)
    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""      # extract_text() returns None on image-only pages
        if not text.strip():                  # skip scanned/empty pages: an empty
            continue                          # chunk would waste an embedding
        documents.append({
            "text": text,
            "metadata": {"source": path.name, "path": str(path),
                         "page": page_number, "type": "pdf"},
        })
    return documents


def load_documents(folder, recursive=True):
    """Load every supported file in a folder into a flat list of Documents."""
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    documents = []
    for path in sorted(folder.glob(pattern)):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT_SUFFIXES:
            documents.extend(load_txt(path))
        elif suffix in SUPPORTED_PDF_SUFFIXES:
            documents.extend(load_pdf(path))
        # anything else (images, .docx, .csv) is silently skipped - extending
        # this dispatch is one of the Session 1 challenges
    return documents


# ===========================================================================
# 2. Text cleaning
# ===========================================================================

# Lines that carry no information but would otherwise become chunk text.
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),  # PDF footers
    re.compile(r"^\s*[-=_*~]{3,}\s*$"),                              # ASCII rules
    re.compile(r"^\s*\d+\s*$"),                                      # bare page numbers
]


def is_boilerplate(line):
    return any(p.match(line) for p in BOILERPLATE_PATTERNS)


def clean_text(text, join_wrapped_lines=True, drop_boilerplate=True):
    """Normalise raw extracted text into something worth embedding.

    join_wrapped_lines=True rebuilds paragraphs from PDF text, where every
    visual line ends with a newline. Set it to False for documents whose
    line breaks are meaningful (code samples, tables, YAML).
    """
    # 1. Unicode normalisation: turns fancy quotes, ligatures and non-breaking
    #    spaces into their plain ASCII-ish equivalents so "don't" and "don't"
    #    do not embed as two different words.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")                      # non-breaking space
    text = text.replace("\r\n", "\n").replace("\r", "\n")     # Windows / old Mac line endings

    # 2. Repair words hyphenated across a line break ("atten-\ntion" -> "attention").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 3. Drop boilerplate lines and squeeze runs of spaces/tabs.
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if drop_boilerplate and is_boilerplate(line):
            continue
        lines.append(line)
    text = "\n".join(lines)

    # 4. Rebuild paragraphs: a single newline is a wrapped line, a blank line is
    #    a real paragraph break. Placeholder trick keeps the two apart.
    if join_wrapped_lines:
        text = re.sub(r"\n{2,}", "<PARA>", text)
        text = text.replace("\n", " ")
        text = text.replace("<PARA>", "\n\n")

    # 5. Collapse leftover whitespace and blank-line runs.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_documents(documents, **kwargs):
    """Apply clean_text to every Document, dropping any that end up empty."""
    cleaned = []
    for doc in documents:
        text = clean_text(doc["text"], **kwargs)
        if text:
            cleaned.append({"text": text, "metadata": dict(doc["metadata"])})
    return cleaned


# ===========================================================================
# 3. Chunking
# ===========================================================================

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into fixed-size character windows that overlap.

    The window advances by (chunk_size - chunk_overlap) characters, so every
    chunk repeats the last `chunk_overlap` characters of the previous one. That
    repetition is what stops a sentence sitting on a boundary from being cut in
    half and made unretrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        # step would be <= 0 and the loop would never move forward
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start, length = 0, len(text)
    while start < length:
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = end - chunk_overlap
    return chunks


def chunk_text_by_paragraph(text, chunk_size=500, chunk_overlap=50):
    """Structure-aware alternative: pack whole paragraphs up to chunk_size.

    Chunks land on real boundaries instead of mid-word, at the cost of uneven
    chunk sizes. Paragraphs longer than chunk_size fall back to chunk_text.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(chunk_text(paragraph, chunk_size, chunk_overlap))
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            # carry the tail of the previous chunk over as overlap
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail} {paragraph}".strip() if tail else paragraph
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(documents, chunk_size=500, chunk_overlap=50, chunker=chunk_text):
    """Chunk a list of Documents, carrying the metadata down to every chunk."""
    all_chunks = []
    for doc in documents:
        pieces = chunker(doc["text"], chunk_size, chunk_overlap)
        for index, piece in enumerate(pieces):
            metadata = dict(doc["metadata"])          # copy: never mutate the parent
            metadata["chunk_index"] = index
            metadata["n_chunks"] = len(pieces)
            metadata["n_chars"] = len(piece)
            source, page = metadata.get("source", "unknown"), metadata.get("page")
            chunk_id = f"{source}#p{page}#c{index}" if page else f"{source}#c{index}"
            all_chunks.append({"id": chunk_id, "text": piece, "metadata": metadata})
    return all_chunks


def build_chunks(folder, chunk_size=500, chunk_overlap=50, chunker=chunk_text, **clean_kwargs):
    """The whole Session 1 pipeline in one call: folder -> list of chunks."""
    documents = load_documents(folder)
    documents = clean_documents(documents, **clean_kwargs)
    return chunk_documents(documents, chunk_size, chunk_overlap, chunker)


# ===========================================================================
# 4. Persistence
# ===========================================================================

def save_chunks(chunks, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_chunks(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preview(text, width=300):
    """Short one-line preview of a chunk, for printing in notebooks."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width] + " ..."


if __name__ == "__main__":
    paths = project_paths()
    chunks = build_chunks(paths["documents"])
    save_chunks(chunks, paths["chunks"])
    print(f"{len(chunks)} chunks -> {paths['chunks']}")
