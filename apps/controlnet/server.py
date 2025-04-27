import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ENV = os.environ.get("NODE_ENV", "development")

env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file)

if ENV == "production":
    prod_env = BASE_DIR / ".env.production"
    if prod_env.exists():
        load_dotenv(prod_env)
elif ENV == "development" or ENV == "local":
    dev_env = BASE_DIR / ".env.local"
    if dev_env.exists():
        load_dotenv(dev_env)

from app import create_app

app = create_app()

if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)

    port = int(os.environ.get("PORT", 8080))

    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
