from io import BytesIO

import pytest
from src.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_no_file_in_request(client):
    response = client.post("/classify")
    assert response.status_code == 400


def test_no_selected_file(client):
    data = {"file": (BytesIO(b""), "")}  # Empty filename
    response = client.post("/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_dummy_data(client):
    data = {"file": (BytesIO(b"dummy content"), "file.pdf")}
    response = client.post("/classify", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


test_data = (
    ["bank_statement_1.pdf", "bank_statement"],
    ["drivers_license_1.jpg", "drivers_license"],
    ["cash_flow.html", "invoice"],
    ["invoice_2.xlsx", "invoice"],
)


@pytest.mark.parametrize("classifier", ("RandomForest", "TokenFrequency"))
@pytest.mark.parametrize("filename,file_class", test_data)
def test_single_file(client, filename: str, classifier: str, file_class: str):
    data = {}

    with open(f"tests/files/{filename}", "rb") as file_reader:
        data = {"file": (file_reader, filename)}
        response = client.post(
            f"/classify?classifier={classifier}",
            data=data,
            content_type="multipart/form-data",
        )
    print(response.get_json())
    assert response.status_code == 200
    assert response.get_json() == {
        "classified_files": {filename: file_class},
        "failed_files": {},
    }
