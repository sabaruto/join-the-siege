from flask import Flask
from werkzeug.datastructures import FileStorage

server = Flask(__name__)


def verifyRequest():
    pass


def getFileType(file: FileStorage) -> str:
    pass


@server.route("/classify_files", methods=["POST"])
def classify_files_route():
    pass
