import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from routers.router import router
from db.connect import connect_db
from gemini.chat import setup_gemini_chat

load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=True, 
     exposed_headers=["set-cookie", "token"])

app.use(router)


async def init_server():
    try:
        port = os.getenv("SERVER_PORT") or 8000
        await connect_db()
        print("DB Connected")
        await setup_gemini_chat()

        app.run(port=port)
        print(f"Backend Server Started on {port} ...")
    except Exception as err:
        print(err)
        print("Server not started!")

if __name__ == "__main__":
    init_server()