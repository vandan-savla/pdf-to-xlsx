"""Public facade: PDF bytes in, formatted Excel workbook out."""

from __future__ import annotations

import io
import logging
from typing import List, Optional

import pandas as pd
from pypdf import PdfReader, PdfWriter

from excel_writer import build_workbook, clean_headers
from ocr_engine import OCREngine
from table_engine import Document, extract_document

logger = logging.getLogger(__name__)


class PasswordRequired(ValueError):
    """The PDF is encrypted and no usable password was supplied."""


class NoTablesFound(ValueError):
    """Nothing table-shaped could be recovered from the document."""


class PDFExtractor:
    """Extracts tables from any PDF -- digital or scanned -- into one workbook.

    A single :class:`OCREngine` is shared across requests so the OCR models are
    loaded at most once per process, and only if a scanned page shows up.
    """

    def __init__(self, ocr_backend: Optional[str] = None):
        self._ocr = OCREngine(ocr_backend)

    # -- decryption ---------------------------------------------------------

    def prepare_pdf_bytes(self, content: bytes, password: Optional[str] = None) -> bytes:
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not read the PDF: {exc}") from exc

        if not reader.is_encrypted:
            return content

        if not password:
            raise PasswordRequired("This PDF is password protected. Please supply the password.")
        if reader.decrypt(password) == 0:
            raise PasswordRequired("Incorrect PDF password.")

        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()

    # -- extraction ---------------------------------------------------------

    def analyse(self, pdf_bytes: bytes) -> Document:
        return extract_document(pdf_bytes, ocr=self._ocr)

    def to_excel_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> bytes:
        document = self.analyse(pdf_bytes)
        if not document.tables:
            raise NoTablesFound(
                "No tables could be detected in this PDF. "
                "If it is a scan, try a higher-quality copy."
            )
        return build_workbook(document, filename=filename)

    # -- convenience --------------------------------------------------------

    def to_dataframes(self, pdf_bytes: bytes) -> List[pd.DataFrame]:
        """Same tables as :meth:`to_excel_bytes`, handy for scripting or tests."""
        frames = []
        for table in self.analyse(pdf_bytes).tables:
            n_cols = table.n_cols
            header = clean_headers(table.header, n_cols) if table.has_header else [
                f"Column {i + 1}" for i in range(n_cols)
            ]
            body = [list(row) + [""] * (n_cols - len(row)) for row in table.body]
            frames.append(pd.DataFrame(body, columns=header))
        return frames
