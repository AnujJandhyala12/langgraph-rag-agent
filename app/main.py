from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.graph import build_graph
from app.tools import credit_risk_tool
from app.rag import ingest_document, SUPPORTED_EXTENSIONS
import re
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os



app = FastAPI(title="Credit Risk RAG Agent")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
graph = build_graph()


class QueryRequest(BaseModel):
    query: str


class ApplicantRequest(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    dti: float


def clean_response(text: str) -> str:
    text = text.replace("\\n", "\n")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"#{1,3} ", "", text)
    return text.strip()

@app.get("/")
def serve_ui():
    return FileResponse("app/dashboard.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")              # ✅ for text questions
async def query_handler(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = graph.invoke({"query": request.query})
        if result is None:
            raise HTTPException(status_code=500, detail="Graph returned None")
        return {
            "query": request.query,
            "response": clean_response(result.get("response", "No response")),  # ✅ must be here
}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")            # ✅ for loan number predictions
def predict(request: ApplicantRequest):
    try:
        applicant_dict = {
            "loan_amnt": request.loan_amnt,
            "int_rate": request.int_rate,
            "annual_inc": request.annual_inc,
            "dti": request.dti,
        }
        result = credit_risk_tool(applicant_dict)
        return {"result": clean_response(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    # ✅ Sanitize filename to prevent path traversal
    filename = os.path.basename(file.filename or "")
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or filename}'. Supported: {supported}",
        )

    try:
        # Save uploaded file to data/ folder
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run ingestion pipeline on the new file
        chunks = ingest_document(file_path)

        return {
            "message": f"{filename} ingested successfully",
            "filename": filename,
            "chunks": chunks,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))