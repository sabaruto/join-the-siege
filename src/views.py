from flask import Request, request, jsonify
from werkzeug.datastructures import FileStorage

from src.classifier import retrieve_classifier
from src.extractor import extract


def verify_request(req: Request):
    if "file" not in request.files:
        raise ValueError("No file part in the request")

    file_list = req.files.getlist("file")
    if len(file_list) == 0:
        return ValueError("No selected file")


def classify_files(file_list: list[FileStorage], classifier_name: str):
    return_dict = {"classified_files": {}, "failed_files": {}}
    classifier = retrieve_classifier(classifier_name)

    for file in file_list:
        try:
            file_contents = extract(file)
        except ValueError as e:
            return_dict["failed_files"][file.filename] = f"File type not allowed {e}"
            continue
        except Exception as e:
            return_dict["failed_files"][
                file.filename
            ] = f"Unable to extract file text: {e}"
            continue

        file_class = classifier.classify(file_contents)
        return_dict["classified_files"][file.filename] = file_class
    return return_dict


def classify_view():
    try:
        verify_request(request)
    except ValueError as e:
        return jsonify({"error": f"{e}"}), 400

    classifier_name = request.args.get("classifier", "RandomForest")
    file_list = request.files.getlist("file")
    return_dict = classify_files(file_list, classifier_name)

    if len(return_dict["failed_files"]) > 0:
        return jsonify(return_dict), 400
    else:
        return jsonify(return_dict), 200
