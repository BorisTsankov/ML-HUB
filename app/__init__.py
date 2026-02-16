from flask import Flask

def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    from .routes import hub_bp
    app.register_blueprint(hub_bp)

    return app
