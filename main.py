import logging
import re
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from extractor import NoTablesFound, PasswordRequired, PDFExtractor

logging.basicConfig(level=logging.INFO)

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

app = FastAPI(title="PDF Table Extractor")
extractor = PDFExtractor()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _page(name: str) -> Response:
    with open(f"static/{name}", "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/html")


@app.get("/")
async def read_landing():
    return _page("index.html")


@app.get("/app")
async def read_app():
    return _page("app.html")


def _safe_stem(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", (filename or "document").rsplit(".", 1)[0])
    return stem.strip("._-")[:80] or "document"


@app.post("/extract-xlsx")
async def extract_xlsx(file: UploadFile = File(...), password: Optional[str] = Form(None)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")

    try:
        pdf_bytes = extractor.prepare_pdf_bytes(content, password)
        workbook = extractor.to_excel_bytes(pdf_bytes, filename=file.filename)
    except PasswordRequired as exc:
        raise HTTPException(401, str(exc)) from exc
    except NoTablesFound as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logging.exception("Extraction failed for %s", file.filename)
        raise HTTPException(500, f"Extraction failed: {exc}") from exc

    return Response(
        content=workbook,
        media_type=XLSX_MEDIA_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{_safe_stem(file.filename)}.xlsx"'
        },
    )
