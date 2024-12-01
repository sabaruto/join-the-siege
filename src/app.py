from src.views import classify_view
from flask import Flask

app = Flask(__name__)

app.add_url_rule("/classify", view_func=classify_view, methods=["POST"])


if __name__ == "__main__":
    app.run()
