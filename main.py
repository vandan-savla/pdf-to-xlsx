import io
import pdfplumber
import pandas as pd
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader, PdfWriter
from pypdf.errors import FileNotDecryptedError

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def read_landing():
    with open("static/landing.html", "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/html")

@app.get("/app")
async def read_app():
    with open("static/app.html", "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/html")

def prepare_pdf_bytes(content: bytes, password: Optional[str] = None) -> bytes:
    """Return bytes of a (possibly decrypted) PDF ready for pdfplumber.

    If the PDF is encrypted and no password is provided, raises HTTPException 401.
    If a password is provided but incorrect, raises HTTPException 401.
    """
    stream = io.BytesIO(content)
    try:
        reader = PdfReader(stream)
    except Exception as e:
        raise Exception(f"Failed to read PDF: {e}")

    if getattr(reader, "is_encrypted", False):
        if not password:
            raise Exception("PDF is password-protected. Provide 'password' form field.")
   
        res = reader.decrypt(password)
        if res == 0:
            raise Exception("Incorrect PDF password")

        # write decrypted PDF to bytes
        writer = PdfWriter()
        for p in reader.pages:
            writer.add_page(p)
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()

    return content


@app.post("/extract-xlsx")
async def extract_tables(file: UploadFile = File(...), password: Optional[str] = Form(None)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        content = await file.read()

        pdf_bytes = prepare_pdf_bytes(content, password)
        pdf_file = io.BytesIO(pdf_bytes)
        
        all_data = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    if table and len(table) > 0:
                        # Just append all rows from the table
                        all_data.extend(table)
        
        if not all_data:
            raise HTTPException(status_code=404, detail="No tables found in PDF")
        
        # Create DataFrame from all merged data
        # First row is header, rest are data
        df = pd.DataFrame(all_data[1:], columns=all_data[0])
        
        # Write to Excel
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        
        return Response(
            content=excel_buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={file.filename.split('.')[0]}.xlsx"}
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-csv")
async def extract_tables(file: UploadFile = File(...), password: Optional[str] = Form(None)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        content = await file.read()
        pdf_bytes = prepare_pdf_bytes(content, password)
        pdf_file = io.BytesIO(pdf_bytes)
        
        all_data = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    if table and len(table) > 0:
                        # Just append all rows from the table
                        all_data.extend(table)
        
        if not all_data:
            raise HTTPException(status_code=404, detail="No tables found in PDF")
        
        # Create DataFrame from all merged data
        # First row is header, rest are data
        df = pd.DataFrame(all_data[1:], columns=all_data[0])
        
        # Write to CSV
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return Response(
            content=csv_buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={file.filename.split('.')[0]}.csv"}
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
