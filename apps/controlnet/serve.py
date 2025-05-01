from app import create_app

# Create the Flask app
app = create_app()

if __name__ == "__main__":
    # When run directly, start the server
    app.run(host="0.0.0.0", port=8080)