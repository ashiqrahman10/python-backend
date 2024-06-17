import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from uuid import uuid4
import asyncio
from websockets import connect
from urllib.parse import urlencode
from datetime import datetime
import aiohttp
import google.generativeai as gen_ai
from flask_mongoengine import MongoEngine
from functools import wraps
from mongoengine.connection import disconnect
from json import JSONEncoder





load_dotenv()

# --- Database Setup ---
db = MongoEngine()

def connect_db():
    db.connect(host=os.getenv("MONGO_URI"))

# --- Database Models ---

class User(db.Document):
    email = db.StringField(required=True, unique=True)
    id = db.StringField(required=True)
    lastmail = db.DateTimeField(default=None)
    totalmail = db.IntField(default=0)

    def to_dict(self):
        return {
            "email": self.email,
            "id": self.id,
            "lastmail": self.lastmail,
            "totalmail": self.totalmail,
        }


class Report(db.Document):
    userId = db.StringField(required=True)
    score = db.IntField()
    keywords = db.ListField(db.StringField())
    analysis = db.StringField()
    timestamp = db.DateTimeField(default=datetime.now)

    def to_dict(self):
        return {
            "userId": self.userId,
            "score": self.score,
            "keywords": self.keywords,
            "analysis": self.analysis,
            "timestamp": self.timestamp,
        }


class ChatHist(db.Document):
    userId = db.StringField(required=True)
    timestamp = db.DateTimeField(default=datetime.now)
    prompt = db.StringField(required=True)
    response = db.StringField(required=True)

    def to_dict(self):
        return {
            "userId": self.userId,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "response": self.response,
        }

# --- Gemini Setup ---

# MODEL_NAME = "gemini-pro"
API_KEY = os.getenv("GEMINI_KEY")

generation_config = {
    "temperature": 0.9,
    "top_k": 1,
    "top_p": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

gemini_model = None

async def setup_gemini_chat():
    global gemini_model
    gen_ai.configure(api_key = os.getenv("GEMINI_KEY"))
    gemini_model = gen_ai.GenerativeModel(model_name=os.getenv("MODEL_NAME"), safety_settings=safety_settings, generation_config=generation_config)


# --- Gemini Prompts ---

analysis_report_prompt = (
    "Make a report or gist of the mental health of the user based on his previous chats. It's length will be 50 to 150 words aprox. Use English language strictly, not even any words of other language. Provide keypoints [Observations, Potential Underlying Issues, Concerns, Recommendations, Overall]"
)
analysis_score_prompt = "Rate the menatal health of the user in a scale of 1 to 10 where 1 is best and 10 is worst based on the previous chats from the user. Just reply the number in the scale 1 to 10, no other things. You are strictly forbidden to reply any other thing than a number."
analysis_keywords_prompt = "Extract keywords from the previous chats of the user that can define its ongoing difficulties and mental health. Use English language strictly, not even any words of other language. You are strictly forbade to use special characters such as asteric(*), dash(-). List the keywords separated by a newline character (\\n). You are strictly forbidden to reply any other thing like word,sentence,character,special characters except keywords. Extract 5 to 10 keywords."

# --- Gemini Chat History ---

init_hist = [
    {
        "role": "user",
        "parts": [
            {
                "text": "Suppose you a mental help advisor AI, instead of Gemini AI. You suggest advice for your patients. Answer like e humanoid chat assistant with a cheerful tone. Only use English language. Reply in aprox 150 words.",
            },
        ],
    },
    {
        "role": "model",
        "parts": [{"text": "Sure."}],
    },
]

def start_gemini_chat(history: list = None):
    if history is None:
        history = []
    return gemini_model.start_chat(
        generation_config=generation_config,
        safety_settings=safety_settings,
        history=[*init_hist, *history],
    )

# --- Firebase Setup ---

import firebase_admin
from firebase_admin import credentials

# credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
cred = credentials.Certificate("firebase-key.json")

firebase_admin.initialize_app(cred)

async def decode_auth_token(token: str) -> str | None:
    try:
        id_token = token.split(" ")[1]
        decoded_token = await firebase_admin.auth().verify_id_token(id_token)
        email = decoded_token.get("email")
        return email
    except Exception as error:
        print(error)
        return None

# --- Flask App Setup ---

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=True,
     exposed_headers=["set-cookie", "token"])
app.config['MONGODB_SETTINGS'] = {
    'db': 'your_database_name',
    'host': os.getenv("MONGO_URI")
}
db.init_app(app)

# --- Middleware ---

def user_middleware(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        user_id = request.cookies.get("userid")
        if user_id and user_id.strip() != "":
            request.userId = user_id
        else:
            user_id = str(uuid4())
            request.userId = user_id
            resp = await func(*args, **kwargs)
            resp = make_response(resp)
            resp.set_cookie(
                "userid",
                user_id,
                max_age=1209600000,  # 14 * 24 * 60 * 60 * 1000 -> 14days
                httponly=True,
                samesite="None",
                secure=True,
            )
            return resp
        return await func(*args, **kwargs)

    return wrapper

# --- Controllers ---

# --- User Controllers ---

@app.route("/signupWithGoogle", methods=["POST"])
async def signup_with_google():
    try:
        token = request.headers.get("token")
        email = await decode_auth_token(token)
        print(email)
        if not email:
            return jsonify({"message": "Invalid Access Token"}), 401

        data = await User.find_one({"email": email})
        if request.cookies.get("userid"):
            # chat already done
            if not data:
                # user not created yet
                user_id = request.cookies.get("userid")
                user = await User.create({
                    "id": user_id,
                    "email": email,
                })
                return jsonify({"data": user.to_dict()}), 200
            else:
                # user already created
                if data.id:
                    resp = make_response(jsonify({"data": data.to_dict()}), 200)
                    resp.set_cookie(
                        "userid",
                        data.id,
                        max_age=1209600000,  # 14 * 24 * 60 * 60 * 1000 -> 14days
                        httponly=True,
                        samesite="None",
                        secure=True,
                    )
                    return resp
        else:
            if not data:
                # user not created yet
                user_id = str(uuid4())
                resp = make_response(jsonify({"Account Created"}), 200)
                resp.set_cookie(
                    "userid",
                    user_id,
                    max_age=1209600000,  # 14 * 24 * 60 * 60 * 1000 -> 14days
                    httponly=True,
                    samesite="None",
                    secure=True,
                )
                user = await User.create({
                    "id": user_id,
                    "email": email,
                })
                return resp
            else:
                # user already created
                if data.id:
                    resp = make_response(jsonify({"data": data.to_dict()}), 200)
                    resp.set_cookie(
                        "userid",
                        data.id,
                        max_age=1209600000,  # 14 * 24 * 60 * 60 * 1000 -> 14days
                        httponly=True,
                        samesite="None",
                        secure=True,
                    )
                    return resp
    except Exception as error:
        return jsonify({"message": "Invalid Access Token"}), 401


@app.route("/signup", methods=["POST"])
async def signup():
    try:
        token = request.headers.get("token")
        email = await decode_auth_token(token)
        print(email)
        if not email:
            return jsonify({"message": "Invalid Access Token"}), 401

        if request.cookies.get("userid"):
            # chat already done
            user_id = request.cookies.get("userid")

            # create user account
            user = await User.create({
                "id": user_id,
                "email": email,
            })

            return jsonify("Account Created"), 200
        else:
            # chat not done yet
            # genereate the uuid and return a cookie

            user_id = str(uuid4())

            # check this if cookie is being set or not
            resp = make_response(jsonify("Account Created"), 200)
            resp.set_cookie(
                "userid",
                user_id,
                max_age=1209600000,  # 14 * 24 * 60 * 60 * 1000 -> 14days
                httponly=True,
                samesite="None",
                secure=True,
            )
            user = await User.create({
                "id": user_id,
                "email": email,
            })
            return resp
    except Exception as error:
        print(error)
        return jsonify({"message": "Invalid Access Token"}), 401


@app.route("/login", methods=["POST"])
async def login():
    try:
        email = await decode_auth_token(request.headers.get("token"))
        if not email:
            return jsonify({"message": "Invalid Access Token"}), 401
        # get Data from email from database
        data = await User.find_one({"email": email})
        if data and data.id:
            resp = make_response(jsonify({"data": data.to_dict()}), 200)
            resp.set_cookie(
                "userid",
                data.id,
                max_age=1209600000,  # 14 * 24 * 60 * 60 * 1000 -> 14days
                httponly=True,
                samesite="None",
                secure=True,
            )
            return resp

        return jsonify({"message": "User not found"}), 404
    except Exception:
        return jsonify({"message": "Invalid Access Token"}), 401


@app.route("/isUser", methods=["GET"])
async def is_user():
    try:
        if request.cookies.get("userid"):
            user_id = request.cookies.get("userid")
            user = await User.find({"id": user_id})
            if user:
                return jsonify({"message": "User validated"}), 200
            else:
                return jsonify({"error": "Logged Out"}), 401
        else:
            return jsonify({"error": "Logged Out"}), 401
    except Exception as error:
        print(error)
        return jsonify({"error": "Logged Out"}), 401


@app.route("/logout", methods=["GET"])
async def logout():
    if not request.cookies.get("userid"):
        return jsonify({"Error": "UserId not found"}), 401
    resp = make_response(jsonify({"msg": "loggedout"}), 200)
    resp.set_cookie(
        "userid",
        "",
        expires=0,
        httponly=True,
        samesite="None",
        secure=True,
    )
    return resp

# --- Analysis Controllers ---

@app.route("/analysis", methods=["GET"])
@user_middleware
async def do_analysis():
    try:
        user_id = request.userId
        if not user_id:
            return jsonify({"Error": "UserId not found"}), 401

        analysis = await generate_analysis(user_id)

        if analysis.get("info") == "nodata":
            return jsonify({"msg": "nochatdata"}), 200

        report_data = await Report.create(
            {
                "userId": user_id,
                "keywords": analysis.get("keywords"),
                "analysis": analysis.get("report"),
                "score": analysis.get("score"),
            }
        )
        try:
            user = await User.find_one({"id": user_id})
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{os.getenv('EMAIL_API_URL')}/welcomeEmail",
                    json={
                        "emailId": user.email,
                        "score": analysis.get("score"),
                        "analysis": analysis.get("report"),
                        "keywords": analysis.get("keywords"),
                    },
                ) as response:
                    if response.status != 200:
                        print("error sending the message")
        except Exception as error:
            print(error)
            print("error sending the message")
        return (
            jsonify({"data": report_data.to_dict()}),
            200,
        )
    except Exception as error:
        print(error)
        return jsonify({"msg": "Internal Server Error"}), 500


async def generate_analysis(user_id: str) -> dict:
    try:
        if not user_id:
            # through err
            return {}
        found_hist = await ChatHist.find({"userId": user_id}).sort("timestamp", 1)

        if not found_hist:
            return {"info": "nodata"}

        found_hist_for_gemini = []
        for conv in found_hist:
            found_hist_for_gemini.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": conv.prompt,
                        },
                    ],
                }
            )
            found_hist_for_gemini.append(
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": conv.response,
                        },
                    ],
                }
            )

        # generate report
        chat = start_gemini_chat(found_hist_for_gemini)
        result = await chat.send_message(analysis_report_prompt)
        response = await result.response
        report = response.text()

        # generate score
        chat = start_gemini_chat(found_hist_for_gemini)
        result = await chat.send_message(analysis_score_prompt)
        response = await result.response
        score = int(response.text())

        # generate keywords
        chat = start_gemini_chat(found_hist_for_gemini)
        result = await chat.send_message(analysis_keywords_prompt)
        response = await result.response
        keywords_resp = response.text()
        keywords = [
            kw.strip()
            for kw in keywords_resp.replace("[^a-zA-Z0-9 \n]", "").strip().split("\n")
            if kw.strip().lower()
            not in ["keyword", "keywords", ""]
        ]
        # console.log(keywords);

        return {"report": report, "score": score, "keywords": keywords}
    except Exception as error:
        print(error)
        return {}


@app.route("/fetchanalysis", methods=["GET"])
@user_middleware
async def get_analysis():
    try:
        if not request.userId:
            return jsonify({"msg": "UserId not found"}), 401
        user_id = request.userId

        reports = (
            await Report.find({"userId": user_id})
            .sort("timestamp", -1)
            .to_list()
        )

        return jsonify({"data": [report.to_dict() for report in reports]}), 200
    except Exception as error:
        print(error)
        return jsonify({"msg": "Internal Server Error"}), 500

# --- Chat Controllers ---

@app.route("/chat", methods=["GET"])
@user_middleware
async def connect_with_chatbot():
    try:
        user_id = request.userId
        if not user_id:
            return jsonify({"Error": "UserId not found"}), 401
        found_hist = await ChatHist.find({"userId": user_id}).sort("timestamp", 1)

        found_hist_for_gemini = []
        for conv in found_hist:
            found_hist_for_gemini.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": conv.prompt,
                        },
                    ],
                }
            )
            found_hist_for_gemini.append(
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": conv.response,
                        },
                    ],
                }
            )

        room_id = str(uuid4())
        params = {"id": room_id, "isServer": True}
        websocketserver_link = (
            f"{os.getenv('WEBSOCKET_SERVER')}?{urlencode(params)}"
        )

        async def websocket_handler(uri):
            async with connect(uri) as websocket:
                await websocket.send(
                    '{"type": "server:connected"}'
                )
                async for message in websocket:
                    try:
                        data = eval(message)
                        if data.get("type") == "client:chathist":
                            await websocket.send(
                                '{"type": "server:chathist", "data": '
                                f"{[hist.to_dict() for hist in found_hist]}}}"
                            )
                        elif data.get("type") == "client:prompt":
                            if not data.get("prompt"):
                                # throw err
                                return

                            # Prompt by the user sent to gemini
                            chat = start_gemini_chat(found_hist_for_gemini)

                            result = await chat.send_message_stream(data.get("prompt"))
                            resp_text = ""
                            await websocket.send('{"type": "server:response:start"}')
                            async for chunk in result.stream:
                                chunk_text = chunk.text()
                                await websocket.send(
                                    '{"type": "server:response:chunk", "chunk": '
                                    f'"{chunk_text}"}}'
                                )
                                resp_text += chunk_text
                            await websocket.send('{"type": "server:response:end"}')
                            # should be stored in the db
                            await ChatHist.create(
                                {
                                    "userId": user_id,
                                    "prompt": data.get("prompt"),
                                    "response": resp_text,
                                }
                            )
                    except Exception as error:
                        print(error)
        asyncio.create_task(websocket_handler(websocketserver_link))
        return jsonify({"chatId": room_id}), 200
    except Exception as error:
        print(error)
        return jsonify({"message": "Websocket connection error"}), 404


# --- Other Routes ---

@app.route("/cron")
async def cron_job():
    return jsonify({"message": "hello"}), 200

# --- Main App Execution ---

async def init_server():
    try:
        port = os.getenv("SERVER_PORT") or 8000
        connect_db()
        print("DB Connected")
        await setup_gemini_chat()

        app.run(port=port)
        print(f"Backend Server Started on {port} ...")
    except Exception as err:
        print(err)
        print("Server not started!")

if __name__ == "__main__":
    asyncio.run(init_server(), debug=True) 