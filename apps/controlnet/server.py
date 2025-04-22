from app import create_app
import os

app = create_app()

if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)

    port = int(os.environ.get("PORT", 8080))

    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
