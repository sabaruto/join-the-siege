from io import BytesIO

import pytest
from src.backend import server


@pytest.fixture
def client():
    server.config["TESTING"] = True
    with server.test_client() as client:
        yield client


def test_no_file_in_request(client):
    response = client.post("/classify_files")
    assert response.status_code == 400


def test_no_selected_file(client):
    data = {"file": (BytesIO(b""), "")}  # Empty filename
    response = client.post(
        "/classify_files", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 400


def test_dummy_data(client, mocker):
    mocker.patch("src.backend.classify_files_route", return_value="test_class")

    data = {"file": (BytesIO(b"dummy content"), "file.pdf")}
    response = client.post(
        "/classify_files", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert response.get_json() == {"file_class": "test_class"}


test_data = (
    ["bank_statement_1.pdf", "bank_statement"],
    ["drivers_license_1.jpg", "drivers_license"],
    ["cash_flow.html", "cash_flow"],
)


@pytest.mark.parametrize("filename,file_class", test_data)
def test_single_file(client, filename: str, file_class: str):
    data = {}

    with open(f"tests/files/{filename}", "rb") as file_reader:
        data = {"file": (file_reader.read(), filename)}

    response = client.post(
        "/classify_files", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert response.get_json() == {"file_class": file_class}
