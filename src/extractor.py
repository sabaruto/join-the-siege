from abc import ABC, abstractmethod
from werkzeug.datastructures import FileStorage
from pypdf import PdfReader
from PIL import Image

import docx2txt
import pandas as pd
import pytesseract
import re
import mimetypes


class Extractor(ABC):
    @abstractmethod
    def extract(self, file: FileStorage) -> any:
        pass


class HTMLToTextExtractor(Extractor):
    def extract(self, file: FileStorage) -> str:
        file_contents = file.stream.read().decode()
        found_text = re.findall(r">([^<\n]+)<", file_contents)
        return "\n".join(found_text)


class PDFToTextExtractor(Extractor):
    def extract(self, file: FileStorage) -> str:
        pdf_reader = PdfReader(file.stream)

        found_text = ""
        for page_index in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_index]
            found_text += page.extract_text()
        return found_text


class XLSXToTextExtractor(Extractor):
    def extract(self, file: FileStorage) -> str:
        xlsx_df = pd.read_excel(file.stream)
        return xlsx_df.to_string()


class CSVToTextExtractor(Extractor):
    def extract(self, file: FileStorage) -> str:
        csv_df = pd.read_csv(file.stream)
        return csv_df.to_string()


class WordToTextExtractor(Extractor):
    def extract(self, file: FileStorage) -> str:
        return docx2txt.process(file.stream)


class ImageToTextExtractor(Extractor):
    def extract(self, file: FileStorage) -> str:
        image = Image.open(file.stream)

        found_text = pytesseract.image_to_string(image, lang="eng", timeout=60)
        with open("check.txt", "w", encoding="utf-8") as file:
            file.write(found_text)

        return found_text


def chooseExtractor(file: FileStorage) -> Extractor:
    mimetype = file.mimetype

    if mimetype == "":
        print(file.filename)
        mimetype, _ = mimetypes.guess_type(file.filename)

    match mimetype:
        case "text/html":
            return HTMLToTextExtractor()
        case "application/pdf":
            return PDFToTextExtractor()
        case "application/xlsx":
            return XLSXToTextExtractor()
        case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return WordToTextExtractor()
        case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            return XLSXToTextExtractor()
        case "text/csv":
            return CSVToTextExtractor()
        case "image/png":
            return ImageToTextExtractor()
        case "image/avif":
            return ImageToTextExtractor()
        case "image/bmp":
            return ImageToTextExtractor()
        case "image/jpeg":
            return ImageToTextExtractor()
        case "image/webp":
            return ImageToTextExtractor()
        case _:
            raise ValueError(f"Passed file type {mimetype} is not allowed")


def extract(file: FileStorage) -> str:
    extractor = chooseExtractor(file)
    text = extractor.extract(file)
    text = re.sub(r"[ \n\t]+", " ", text)
    return text
