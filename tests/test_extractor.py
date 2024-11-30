from src.extractor import (
    HTMLToTextExtractor,
    ImageToTextExtractor,
    PDFToTextExtractor,
    WordToTextExtractor,
    XLSXToTextExtractor,
    chooseExtractor,
)
from werkzeug.datastructures import FileStorage

import pytest


@pytest.mark.parametrize("filename", ["audio.flac", "video.mp4", "db.sql"])
def test_disallowed_file(filename: str):
    file = FileStorage(filename=f"tests/files/{filename}")

    with pytest.raises(ValueError):
        chooseExtractor(file)


test_data = [
    ("invoice_1.pdf", PDFToTextExtractor.__name__),
    ("drivers_license_1.jpg", ImageToTextExtractor.__name__),
    ("bank_statement_1.pdf", PDFToTextExtractor.__name__),
    ("cash_flow.html", HTMLToTextExtractor.__name__),
    ("invoice_2.xlsx", XLSXToTextExtractor.__name__),
    ("bank_statement_2.docx", WordToTextExtractor.__name__),
]


@pytest.mark.parametrize("filename,extractor_class", test_data)
def test_allowed_file(filename: str, extractor_class: str):
    file = FileStorage(open(f"tests/files/{filename}", "rb"))
    extractor = chooseExtractor(file)

    assert extractor.__class__.__name__ is extractor_class

    file_contents = extractor.extract(file)
    assert type(file_contents) is str
    assert len(file_contents) > 0
