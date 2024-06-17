import os
import uuid
import datetime
from functools import wraps
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from pymongo import MongoClient
from firebase_admin import initialize_app, auth, credentials
from google.generativeai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from string_similarity import compare_two_strings

# --- Constants ---
DB_NAME = "MindCare"
TOTAL_QUESTIONS = 15
DEFAULT_QUESTIONS = [
    {
        "QuestionNo": 1,
        "QuestionText": "How would you describe your overall mood most of the time?",
        "options": [
            "Very positive and optimistic",
            "Generally positive",
            "Neutral or mixed",
            "Frequently sad, anxious, or depressed",
        ],
    },
    {
        "QuestionNo": 2,
        "QuestionText": "Do you find enjoyment in activities that used to bring you pleasure?",
        "options": [
            "Yes, consistently",
            "Sometimes, but less often",
            "Rarely or occasionally",
            "Rarely or never, even in activities I used to enjoy",
        ],
    },
    {
        "QuestionNo": 3,
        "QuestionText": "How well are you sleeping on average?",
        "options": [
            "Well, consistently",
            "Occasionally disrupted but generally good",
            "Poorly or inconsistently",
            "Very poorly or experiencing significant sleep disturbances",
        ],
    },
    {
        "QuestionNo": 4,
        "QuestionText": "How would you rate your energy levels throughout the day?",
        "options": [
            "High and consistent",
            "Moderate and steady",
            "Low or fluctuating",
            "Very low or experiencing extreme fluctuations",
        ],
    },
    {
        "QuestionNo": 5,
        "QuestionText": "Are you experiencing changes in appetite or weight?",
        "options": [
            "No changes",
            "Some changes, but manageable",
            "Significant changes",
            "Drastic changes impacting daily functioning",
        ],
    },
    {
        "QuestionNo": 6,
        "QuestionText": "Do you often find it challenging to concentrate or make decisions?",
        "options": [
            "Rarely or never",
            "Occasionally",
            "Frequently",
            "Constantly, affecting daily tasks and decision-making",
        ],
    },
    {
        "QuestionNo": 7,
        "QuestionText": "How would you describe your social interactions and relationships lately?",
        "options": [
            "Positive and fulfilling",
            "Generally positive with occasional challenges",
            "Strained or isolating",
            "Severely strained, impacting multiple relationships",
        ],
    },
    {
        "QuestionNo": 8,
        "QuestionText": "Do you experience periods of intense worry or fear without an apparent cause?",
        "options": [
            "Rarely or never",
            "Occasionally",
            "Frequently",
            "Almost constantly, interfering with daily life",
        ],
    },
    {
        "QuestionNo": 9,
        "QuestionText": "Have you noticed any changes in your physical health, such as unexplained aches or pains?",
        "options": [
            "No changes",
            "Occasionally",
            "Frequently",
            "Persistent and severe physical health issues",
        ],
    },
    {
        "QuestionNo": 10,
        "QuestionText": "How do you cope with stress on a day-to-day basis?",
        "options": [
            "Effective coping strategies",
            "Some coping mechanisms",
            "Ineffective or maladaptive coping",
            "No effective coping mechanisms, leading to increased distress",
        ],
    },
    {
        "QuestionNo": 11,
        "QuestionText": "Have you had thoughts of self-harm or suicide?",
        "options": [
            "No",
            "Rarely",
            "Occasionally",
            "Frequently or consistently",
        ],
    },
    {
        "QuestionNo": 12,
        "QuestionText": "Do you experience racing thoughts or restlessness?",
        "options": [
            "Rarely or never",
            "Occasionally",
            "Frequently",
            "Almost constantly, affecting daily functioning",
        ],
    },
    {
        "QuestionNo": 13,
        "QuestionText": "How do you handle setbacks or challenges in your life?",
        "options": [
            "Resiliently and effectively",
            "With some difficulty",
            "Poorly or not at all",
            "Overwhelmed, leading to a significant decline in functioning",
        ],
    },
    {
        "QuestionNo": 14,
        "QuestionText": "Are there any specific traumas or major life changes you've experienced recently?",
        "options": [
            "No major traumas or changes",
            "Some moderate changes or challenges",
            "Significant traumas or life-altering events",
            "Severe traumas or multiple major life changes",
        ],
    },
    {
        "QuestionNo": 15,
        "QuestionText": "How would you rate your overall stress level on a scale from 1 to 10, with 10 being the highest?",
        "options": [
            "1-3 (Low stress)",
            "4-6 (Moderate stress)",
            "7-8 (High stress)",
            "9-10 (Severe stress)",
        ],
    },
]
# --- App Configuration ---
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# --- Database Connection ---
mongo_uri = os.environ.get("MONGO_URI")
client = MongoClient(mongo_uri)
db = client[DB_NAME]
user_collection = db["User"]
report_collection = db["Report"]
chat_hist_collection = db["ChatHist"]
question_collection = db["Question"]
appointment_collection = db["AppointmentModel"]
user_response_collection = db["UserResponse"]

# --- Firebase Initialization ---
cred = credentials.Certificate(os.environ.get("FIREBASE_KEY"))  # Replace with your path
initialize_app(cred)

# --- Gemini Setup ---
GEMINI_MODEL_NAME = "gemini-1.5-pro"  # Replace with your desired Gemini model
GEMINI_API_KEY = os.environ.get("GEMINI_KEY")

gemini = GoogleGenerativeAI(GEMINI_API_KEY)
gemini_model = gemini.get_generative_model(model=GEMINI_MODEL_NAME)

GENERATION_CONFIG = {
    "temperature": 0.9,
    "top_k": 1,
    "top_p": 1,
    "max_output_tokens": 2048,
}

SAFETY_SETTINGS = [
    {
        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
]


# --- Helper Functions ---


def generate_uuid():
    return str(uuid.uuid4())


def decode_auth_token(token):
    try:
        id_token = token.split(" ")[1]
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token["email"]
    except Exception as e:
        print(f"Error decoding token: {e}")
        return None


def generate_analysis(user_id):
    try:
        # Fetch chat history
        chat_history = list(
            chat_hist_collection.find({"userId": user_id}).sort("timestamp", 1)
        )

        if not chat_history:
            return {"info": "nodata"}

        # Format chat history for Gemini
        found_hist_for_gemini = []
        for conv in chat_history:
            found_hist_for_gemini.append(
                {"role": "user", "parts": [{"text": conv["prompt"]}]}
            )
            found_hist_for_gemini.append(
                {"role": "model", "parts": [{"text": conv["response"]}]}
            )

        # Generate report
        chat = gemini_model.start_chat(
            history=found_hist_for_gemini,  # Add chat history
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
        )
        report_response = chat.send_message(
            "Make a report or gist of the mental health of the user based on his previous chats. It's length will be 50 to 150 words approx. Use English language strictly, not even any words of other language. Provide keypoints [Observations, Potential Underlying Issues, Concerns, Recommendations, Overall]"
        )
        report = report_response.text

        # Generate score
        score_response = chat.send_message(
            "Rate the menatal health of the user in a scale of 1 to 5 where 1 is best and 10 is worst based on the previous chats from the user. Just reply the number in the scale 1 to 10, no other things. You are strictly forbidden to reply any other thing than a number."
        )
        score = int(score_response.text)

        # Generate keywords
        keywords_response = chat.send_message(
            "Extract keywords from the previous chats of the user that can define its ongoing difficulties and mental health. Use English language strictly, not even any words of other language. You are strictly forbade to use special characters such as asteric(*), dash(-). List the keywords separated by a newline character (\n). You are strictly forbidden to reply any other thing like word,sentence,character,special characters except keywords. Extract 5 to 10 keywords."
        )
        keywords = [
            kw.strip()
            for kw in keywords_response.text.replace("[", "")
            .replace("]", "")
            .split("\n")
            if kw.strip()
        ]

        return {"report": report, "score": score, "keywords": keywords}
    except Exception as error:
        print(f"Error in generate_analysis: {error}")
        return None


def find_closest_match(user_text, backend_options):
    ratings = [
        compare_two_strings(user_text.lower(), option.lower())
        for option in backend_options
    ]
    best_match_index = ratings.index(max(ratings))
    return best_match_index


# --- Decorators ---
def user_middleware(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        user_id = request.cookies.get("userid")
        if user_id and user_id.strip():
            request.user_id = user_id
        else:
            request.user_id = generate_uuid()
            response = make_response(func(*args, **kwargs))
            response.set_cookie(
                "userid",
                request.user_id,
                max_age=1209600,  # 14 days
                httponly=True,
                samesite="None",
                secure=True,
            )
            return response
        return func(*args, **kwargs)

    return wrapper


# --- Authentication Routes ---


@app.route("/signup", methods=["POST"])
@user_middleware
def signup():
    try:
        token = request.headers.get("token")
        email = decode_auth_token(token)
        if not email:
            return jsonify({"message": "Invalid Access Token"}), 401
        if request.user_id:
            user = user_collection.find_one({"id": request.user_id})
            if user:
                return jsonify({"message": "Account already exists"}), 200
            else:
                user_collection.insert_one({"id": request.user_id, "email": email})
                return jsonify({"message": "Account Created"}), 200
    except Exception as e:
        print(f"Error in signup: {e}")
        return jsonify({"message": "Signup failed"}), 500


@app.route("/login", methods=["POST"])
@user_middleware
def login():
    try:
        token = request.headers.get("token")
        email = decode_auth_token(token)
        if not email:
            return jsonify({"message": "Invalid Access Token"}), 401

        user = user_collection.find_one({"email": email})
        if user:
            return jsonify({"data": user}), 200
        else:
            return jsonify({"message": "User not found"}), 404
    except Exception as e:
        print(f"Error in login: {e}")
        return jsonify({"message": "Login failed"}), 500


@app.route("/isUser")
@user_middleware
def is_user():
    try:
        if request.user_id:
            user = user_collection.find_one({"id": request.user_id})
            if user:
                return jsonify({"message": "User validated"}), 200
            else:
                return jsonify({"error": "Logged Out"}), 401
        else:
            return jsonify({"error": "Logged Out"}), 401
    except Exception as e:
        print(f"Error in is_user: {e}")
        return jsonify({"error": "Logged Out"}), 401


@app.route("/logout")
@user_middleware
def logout():
    try:
        response = make_response(jsonify({"msg": "loggedout"}), 200)
        response.set_cookie("userid", "", expires=0)
        return response
    except Exception as e:
        print(f"Error in logout: {e}")
        return jsonify({"message": "Logout failed"}), 500


# --- Chat Route ---
@app.route("/chat", methods=["GET"])
@user_middleware
def connect_with_chatbot():
    return jsonify({"message": "Chat functionality is not implemented"}), 501


# --- Analysis Routes ---


@app.route("/analysis")
@user_middleware
def do_analysis():
    try:
        user_id = request.user_id
        analysis = generate_analysis(user_id)

        if analysis.get("info") == "nodata":
            return jsonify({"msg": "nochatdata"}), 200

        report_data = {
            "userId": user_id,
            "keywords": analysis["keywords"],
            "analysis": analysis["report"],
            "score": analysis["score"],
            "timestamp": datetime.datetime.utcnow(),
        }

        report_collection.insert_one(report_data)

        # (Optional) Send email using another service

        return jsonify({"data": report_data}), 200
    except Exception as error:
        print(f"Error in do_analysis: {error}")
        return jsonify({"msg": "Internal Server Error"}), 500


@app.route("/fetchanalysis")
@user_middleware
def get_analysis():
    try:
        user_id = request.user_id
        reports = list(
            report_collection.find({"userId": user_id}).sort("timestamp", -1)
        )
        return jsonify({"data": reports}), 200
    except Exception as error:
        print(f"Error in get_analysis: {error}")
        return jsonify({"msg": "Internal Server Error"}), 500


# --- User Location Route ---


@app.route("/location", methods=["POST"])
@user_middleware
def doctor_location():
    try:
        latitude = request.json.get("latitude")
        longitude = request.json.get("longitude")

        if not latitude or not longitude:
            return (
                jsonify({"error": "Latitude and longitude are required."}),
                400,
            )

        # Your logic to fetch and return doctor locations
        # based on the provided latitude and longitude

        # Example:
        doctors = [
            {"name": "Dr. Smith", "latitude": latitude + 0.01, "longitude": longitude + 0.01},
            {"name": "Dr. Jones", "latitude": latitude - 0.02, "longitude": longitude - 0.02},
        ]

        return jsonify({"doctors": doctors}), 200
    except Exception as e:
        print(f"Error in doctor_location: {e}")
        return jsonify({"error": "Failed to fetch doctor locations."}), 500


# --- User Assessment Routes ---


@app.route("/user/userAssessment/question", methods=["GET"])
@user_middleware
def get_user_response():
    try:
        user_id = request.user_id
        saved_data = user_response_collection.find_one({"userId": user_id})
        if saved_data:
            return jsonify(saved_data), 200
        else:
            return jsonify({"message": "No data found"}), 404
    except Exception as error:
        print(f"Error in get_user_response: {error}")
        return jsonify({"message": "Failed to retrieve user response"}), 500


@app.route("/user/userAssessment/save", methods=["POST"])
@user_middleware
def save_user_response():
    try:
        data = request.get_json()
        token = data.get("token")
        user_res = data.get("userRes")
        question_name = data.get("QuestionName")
        final_submit = data.get("finalSubmit", False)

        user_id = decode_auth_token(token)
        if not user_id:
            return jsonify({"msg": "Invalid token"}), 401

        if not question_name or not user_res:
            return jsonify({"msg": "Body data fields are empty"}), 400

        # Find the question in the database
        similar_question = question_collection.find_one(
            {"question.QuestionText": question_name},
            {"question.$": 1, "_id": 0},
        )
        if not similar_question:
            return jsonify({"msg": "Question not found"}), 404

        backend_options = similar_question["question"][0]["options"]

        # Find the closest matching option
        match_result = find_closest_match(user_res, backend_options)
        user_selected_option = backend_options[match_result]

        assessment = {"q": question_name, "a": user_selected_option}

        existing_user_response = user_response_collection.find_one({"userId": user_id})
        if existing_user_response is None:
            # Create a new user response document if it doesn't exist
            new_user_response = {
                "userId": user_id,
                "AllAssessments": [{"complete": False, "assessments": [assessment]}],
            }
            user_response_collection.insert_one(new_user_response)
            return jsonify({"msg": "User created and first data pushed"}), 201

        last_assessment = existing_user_response["AllAssessments"][-1]

        if last_assessment["complete"] and len(last_assessment["assessments"]) >= 15:
            # If the last assessment is complete and has 15 assessments,
            # create a new assessment object
            existing_user_response["AllAssessments"].append(
                {"complete": False, "assessments": [assessment]}
            )
        else:
            # Otherwise, append the assessment to the last assessment object
            found = False
            for i, assess in enumerate(last_assessment["assessments"]):
                if assess["q"] == question_name:
                    last_assessment["assessments"][i] = assessment
                    found = True
                    break
            if not found:
                last_assessment["assessments"].append(assessment)

            if len(last_assessment["assessments"]) >= 15 and final_submit:
                last_assessment["complete"] = True

        user_response_collection.update_one(
            {"userId": user_id}, {"$set": existing_user_response}
        )
        return jsonify({"msg": "Data saved successfully"}), 200

    except Exception as e:
        print(f"Error in save_user_response: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/user/userAssessment/submit", methods=["POST"])
@user_middleware
def update_complete_status():
    try:
        data = request.get_json()
        token = data.get("token")
        final_submit = data.get("finalSubmit", False)

        user_id = decode_auth_token(token)
        if not user_id:
            return jsonify({"msg": "Invalid token"}), 401

        existing_user_response = user_response_collection.find_one({"userId": user_id})
        if not existing_user_response:
            return jsonify({"msg": "User not found"}), 404

        last_assessment = existing_user_response["AllAssessments"][-1]

        if len(last_assessment["assessments"]) >= 15 and final_submit:
            last_assessment["complete"] = True
            user_response_collection.update_one(
                {"userId": user_id}, {"$set": existing_user_response}
            )
            return jsonify({"msg": "Complete status updated successfully"}), 200
        else:
            return jsonify({"msg": "Unable to update complete status"}), 400

    except Exception as e:
        print(f"Error in update_complete_status: {e}")
        return jsonify({"error": "Internal server error"}), 500


# --- Admin Assessment Routes ---


@app.route("/admin/assessment/get", methods=["GET"])
def get_user_assessment():
    try:
        get_saved_data = list(
            question_collection.find({"customid": "mindcareAdmin"})
        )

        if not get_saved_data:
            # If no custom data is found, insert default questions
            question_collection.insert_many(
                [
                    {"customid": "mindcareAdmin", "question": DEFAULT_QUESTIONS}
                ]
            )
            get_saved_data = DEFAULT_QUESTIONS
        else:
            get_saved_data = get_saved_data[0]["question"]

        print("Data receive successfully")
        return jsonify({"data": get_saved_data}), 200
    except Exception as error:
        print(error)
        return jsonify({"message": "Data not found"}), 404


@app.route("/admin/assessment/update", methods=["PATCH"])
def update_user_assessment():
    return jsonify({"message": "Not Implemented"}), 501


@app.route("/admin/assessment/delete", methods=["DELETE"])
def delete_user_assessment():
    try:
        question_no = request.json.get("QuestionNo")
        if question_no is None:
            return jsonify({"error": "QuestionNo is required"}), 400

        # Delete the question with the specified QuestionNo
        after_delete = question_collection.update_many(
            {}, {"$pull": {"question": {"QuestionNo": question_no}}}
        )

        # Update the QuestionNo for the remaining questions
        total_doc_no = question_collection.count_documents({})
        for i in range(question_no, total_doc_no):
            response = question_collection.update_many(
                {"question.QuestionNo": i + 1},
                {"$set": {"question.$.QuestionNo": i}},
            )
            print(response)

        print(after_delete)
        return jsonify({"msg": "success"}), 201

    except Exception as error:
        print(f"Error in delete_user_assessment: {error}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/admin/assessment/putdata", methods=["PUT"])
def put_user_assessment():
    try:
        response = request.get_json()
        if not response:
            return jsonify({"message": "Empty data sent"}), 400

        check_data = question_collection.count_documents({})
        if check_data:
            save_data = question_collection.update_one(
                {"customid": "mindcareAdmin"}, {"$push": {"question": response}}
            )
            print(save_data)
        else:
            save_data = question_collection.insert_many(
                [{"customid": "mindcareAdmin", "question": response}]
            )
            print(save_data)

        return jsonify({"msg": "Success"}), 201

    except Exception as error:
        print(f"Error in put_user_assessment: {error}")
        return jsonify({"msg": "Failed to add questions"}), 500


# --- Doctor Appointment Routes ---


@app.route("/dr/appointment/details", methods=["GET"])
def get_appointment_details():
    try:
        dr_token = request.args.get("drToken")
        user_id = request.json.get("userId") if request.json else None
        dr_id = decode_auth_token(dr_token)
        if not dr_id:
            return jsonify({"message": "Invalid token"}), 401

        if not user_id:
            response = appointment_collection.find_one({"drId": dr_id})
            if response:
                return jsonify({"Found_data": response}), 200
            else:
                return jsonify({"No data found": response}), 404

        response = appointment_collection.find_one(
            {"drId": dr_id, "appointmentDetails.userId": user_id},
            {"appointmentDetails.$": 1, "_id": 0},
        )

        if response:
            return (
                jsonify({"Data received": response.get("appointmentDetails", [])}),
                200,
            )
        else:
            return jsonify({"No data found": response}), 404

    except Exception as e:
        print(f"Error in get_appointment_details: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/dr/req/save", methods=["POST"])
def save_request():
    try:
        dr_id = request.json.get("drId")
        patient_id = request.json.get("patientId")
        user_details = request.json.get("userDetails")

        user_id = (
            decode_auth_token(patient_id) if " " in patient_id else patient_id
        )

        if not dr_id or not user_details or not user_id:
            return jsonify({"message": "Request body data is missing"}), 400

        find_response = appointment_collection.find_one({"drId": dr_id})

        if find_response is None:
            saved = appointment_collection.insert_one(
                {
                    "drId": dr_id,
                    "appointmentDetails": [
                        {"userId": user_id, "userDetails": user_details}
                    ],
                }
            )
            print("Data saved:", saved)
            return jsonify({"message": "success"}), 200
        else:
            updated = appointment_collection.update_one(
                {"drId": dr_id},
                {
                    "$push": {
                        "appointmentDetails": {
                            "userId": user_id,
                            "userDetails": user_details,
                        }
                    }
                },
            )

            if updated.modified_count:
                print("Data updated and saved", updated)
                return (
                    jsonify({"message": "Data updated and saved", "data": updated}),
                    200,
                )
            else:
                print("Failed to update data")
                return jsonify({"message": "Failed to update data"}), 500

    except Exception as error:
        print(f"Error in save_request: {error}")
        return jsonify({"message": "Failed to save request"}), 500


@app.route("/dr/req/acceptstatus", methods=["PUT"])
def save_accept_status():
    try:
        dr_token = request.json.get("drToken")
        user_id = request.json.get("userId")
        req_accepted = request.json.get("reqAccepted")

        dr_id = decode_auth_token(dr_token)
        if not dr_id:
            return jsonify({"message": "Invalid token"}), 401

        if not user_id or req_accepted is None:
            return jsonify({"message": "Request body data missing"}), 400

        if req_accepted is False:
            return jsonify({"message": "False is default. Only send true."}), 400

        response = appointment_collection.update_one(
            {"drId": dr_id, "appointmentDetails.userId": user_id},
            {"$set": {"appointmentDetails.$.reqAccepted": req_accepted}},
        )

        if response.matched_count and response.modified_count:
            print("Accept status Data updated successfully")
            return jsonify({"message": "Accept status Data updated successfully"}), 200
        elif response.matched_count and not response.modified_count:
            print("Accept status already true")
            return jsonify({"message": "Accept status already true"}), 400
        else:
            print("User or doctor not found")
            return jsonify({"message": "User or doctor not found"}), 404

    except Exception as error:
        print(f"Error in save_accept_status: {error}")
        return jsonify({"message": "Failed to update accept status"}), 500


@app.route("/dr/appointment/delete", methods=["DELETE"])
def delete_appointment_details():
    try:
        dr_token = request.args.get("drToken")
        user_id = request.args.get("patientId")
        dr_id = decode_auth_token(dr_token)
        if not dr_id:
            return jsonify({"message": "Invalid token"}), 401

        if not user_id:
            return jsonify({"message": "patientId is required"}), 400

        response = appointment_collection.find_one_and_update(
            {"drId": dr_id, "appointmentDetails.userId": user_id},
            {"$pull": {"appointmentDetails": {"userId": user_id}}},
            return_document=True,
        )

        if response:
            if len(response.get("appointmentDetails", [])) == 0:
                appointment_collection.delete_one({"drId": dr_id})
                return (
                    jsonify(
                        {
                            "message": "Data deleted...therefore Dr appointment list got empty and deleted successfully"
                        }
                    ),
                    200,
                )
            else:
                print("Data deleted and data after delete:", response)
                return (
                    jsonify(
                        {
                            "message": "Data deleted and data after delete:",
                            "data": response,
                        }
                    ),
                    200,
                )
        else:
            print("No data found to delete")
            return jsonify({"message": "No data found to delete"}), 404

    except Exception as error:
        print(f"Error in delete_appointment_details: {error}")
        return jsonify({"message": "Failed to delete appointment details"}), 500


# --- AI Assessment Routes ---


@app.route("/ai/assessment/<token>", methods=["GET"])
def get_ai_response(token):
    try:
        user_id = decode_auth_token(token)

        if not user_id:
            return jsonify({"msg": "Invalid token"}), 401

        get_saved_data = user_response_collection.find_one({"userId": user_id})
        if not get_saved_data:
            return jsonify({"message": "User response data not found"}), 404

        last_index = len(get_saved_data["AllAssessments"]) - 1
        last_assessment = get_saved_data["AllAssessments"][last_index]["assessments"]

        # Construct the prompt for the AI model
        predefine_text = (
            "Analysis Request: Given a dataset containing a set of questions and corresponding answers, "
            "analyze the data and provide comprehensive advice. Suggest some exercise that can "
            "refresh my mind. Also, how can I improve my mental health? Please give the data in a "
            "well-structured format."
        )
        prompt = f"Data: {last_assessment} {predefine_text}"

        # Generate response from Gemini
        response = gemini_model.generate_content(
            prompt, generation_config=GENERATION_CONFIG, safety_settings=SAFETY_SETTINGS
        )

        return jsonify(response.text), 200

    except Exception as error:
        print(f"Error in get_ai_response: {error}")
        return jsonify({"message": str(error)}), 500


# --- User Profile Routes ---


@app.route("/user/profile", methods=["GET", "POST", "PATCH", "DELETE"])
@user_middleware
def user_profile():
    # Your user profile logic here
    return jsonify({"message": "User profile functionality is not implemented"}), 501


# --- Doctor Profile Routes ---


@app.route("/doctors/profile", methods=["GET"])
def doctors_profile():
    # Your doctor profile logic here
    return (
        jsonify({"message": "Doctor profile functionality is not implemented"}),
        501,
    )


if __name__ == "__main__":
    app.run(debug=True)