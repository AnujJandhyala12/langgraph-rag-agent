import os
import time
import pandas as pd
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "credit-risk-index"
# ✅ Safety cap for tabular files so a huge dataset can't blow up the index
MAX_ROWS = 500


def _get_embeddings():
    # ✅ Shared embedding model (384-dim MiniLM)
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def _semantic_chunk(texts, embeddings):
    # ✅ Semantic chunking — splits by meaning not character count
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )
    return splitter.create_documents(texts)


def create_retriever():
    # ✅ Load embedding model
    embeddings = _get_embeddings()

    # ✅ Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = INDEX_NAME

    # ✅ Create index only if it doesn't exist
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
        print(f"Created Pinecone index: {index_name}")

    # ✅ Seed the report ONLY when the index is empty — avoids re-embedding
    #    the same document (and duplicating vectors) on every boot.
    vector_count = pc.Index(index_name).describe_index_stats().get("total_vector_count", 0)
    if vector_count == 0:
        loader = TextLoader("data/credit_risk_report.txt")
        docs = loader.load()
        split_docs = _semantic_chunk([doc.page_content for doc in docs], embeddings)
        print(f"Seeding empty index with {len(split_docs)} semantic chunks")
        vectorstore = PineconeVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings,
            index_name=index_name,
        )
    else:
        print(f"Index already populated ({vector_count} vectors) — skipping re-ingest")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )

    # ✅ MMR retrieval — relevance + diversity
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
    )



# ✅ Formats the ingestion pipeline knows how to read
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx", ".xls"}


def _rows_to_documents(df: pd.DataFrame, source: str) -> list[Document]:
    """Turn each (capped) table row into one searchable record."""
    df = df.head(MAX_ROWS)
    docs = []
    for i, row in df.iterrows():
        content = " | ".join(
            f"{col}: {val}" for col, val in row.items() if pd.notna(val)
        )
        if content.strip():
            docs.append(
                Document(page_content=content, metadata={"source": source, "row": int(i)})
            )
    return docs


def ingest_document(file_path: str) -> int:
    """Ingest a new file into the shared Pinecone index and return the chunk count.

    Supports pdf/txt/md/docx (semantic chunking) and csv/xlsx/xls
    (row-per-record, capped at MAX_ROWS rows).
    """
    ext = os.path.splitext(file_path)[1].lower()
    source = os.path.basename(file_path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    embeddings = _get_embeddings()

    if ext in {".csv", ".xlsx", ".xls"}:
        # ✅ Tabular: read only up to MAX_ROWS so a giant file never loads fully
        if ext == ".csv":
            df = pd.read_csv(file_path, nrows=MAX_ROWS)
        else:
            df = pd.read_excel(file_path, nrows=MAX_ROWS)
        split_docs = _rows_to_documents(df, source)
    elif ext in {".txt", ".md", ".pdf", ".docx"}:
        # ✅ Text-like: load then semantic-chunk by meaning
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        split_docs = _semantic_chunk([doc.page_content for doc in docs], embeddings)
        for doc in split_docs:
            doc.metadata["source"] = source

    if not split_docs:
        raise ValueError("No extractable text found in the file")

    # ✅ Add to the same index — immediately queryable by the live retriever
    PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=INDEX_NAME,  # same index, adds to existing
    )

    print(f"✅ Ingested {len(split_docs)} chunks from {source}")
    return len(split_docs)
