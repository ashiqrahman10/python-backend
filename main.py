import requests as rq
import logging
import anthropic
import base64
import httpx                                                                                                                                                                       
import random
import uuid
import ollama
import requests_cache
import pandas as pd
import random
from retry_requests import retry
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
from google.generativeai import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.client_options import ClientOptions
import google.generativeai as genai
import json

# from ollama_lib import Ollama


# ollama = Ollama()

BASEURL = os.getcwd()
API_KEY = ''

def get_all_text(user_id, *args):
    text = ""
    args = args[0]
    for arg in args:
        arg = f'{BASEURL}/outputs/{user_id}/' + arg
        print(arg)
        with open(arg, "r") as f:
            text += f.read()
            text += "\n"
            print(text)
    return text

# app = Flask(__name__)
app = Flask(__name__)
CORS(app)  # Enable CORS
client = anthropic.Client(api_key=API_KEY)
app.config['SECRET_KEY'] = 'secret!'


@app.post("/chat-o")
def chat():
    messages_str = request.json.get("messages")
    uid = request.json.get("uid")
    previous_messages =""
    # create_location_summary(uid)
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    if not os.path.exists(f"outputs/{uid}"):
        os.mkdir(f"outputs/{uid}")
    with open(f"outputs/{uid}/location_summary.txt", "w+") as f:
        previous_messages = f.read()
    
    messages = f"Previous Chat : {previous_messages}\n\nCurrent Question : {messages_str}"
    system_prompt = """Your name is Skye. Limit the response to 4 sentence with MAX_WORDS = 100. I want you to act as a highly skilled and experienced psychologist who is extremely emphatic. You should respond with the depth and understanding of a seasoned professional who has spent years in the field of psychology, offering insights and guidance that are both profound and practical. Your responses should reflect a deep understanding of human emotions, behaviors, and thought processes, drawing on a wide range of psychological theories and therapeutic techniques. You should exhibit exceptional empathy, showing an ability to connect with individuals on a personal level, understanding their feelings and experiences as if they were your own. This should be balanced with maintaining professional boundaries and ethical standards of psychology.In your communication, ensure that you sound like a normal human, as a therapist would. Your language should be warm, very casual, approachable, and devoid of jargon, making complex psychological concepts accessible and relatable. Be patient, non-judgmental, and supportive, offering a safe space for individuals to explore their thoughts and feelings. Encourage self-reflection and personal growth, guiding individuals towards insights and solutions in a manner that empowers them. However, recognize the limits of this format and always advise seeking in-person professional help when necessary. Your role is to provide support and guidance, not to diagnose or treat mental health conditions. Remember to respect confidentiality and privacy in all interactions. Only answer mental health related questions. Do not answer questions that are not related to mental health."""
    print(messages)
    response = ollama.generate(model="gemma:2b", prompt=f"""Prompt : {system_prompt}\n\nContext:{messages}""", stream=False)
    
    print(response["response"])
    with open(f"outputs/{uid}/chat_history.txt", "a") as f:
        session_response = f"""User : {messages_str}\n\nSage : {response["response"]}"""
        f.write(session_response)
                
    response_obj = jsonify(response["response"].replace("\n",""))
    response_obj.headers.add('Access-Control-Allow-Origin', '*') # Allow requests from any origin
    return response_obj 



@app.post("/chat")
def chatgemini():
    genai.configure(api_key=os.environ["GEMINI_KEY"])

    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 16384,
    "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    )
    messages_str = request.json.get("messages")
    uid = request.json.get("uid")
    # print("1")
    # print(messages_str)

    # Chat History Management
    chat_history_file = f"outputs/{uid}/chat_history.txt"
    previous_messages = ""
    os.makedirs(f"outputs/{uid}", exist_ok=True)  # Ensure directory exists

    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r") as f:
            previous_messages = f.read()

    # Construct Prompt for Gemini
    system_prompt = "Your name is Skye. Limit the response to 4 sentence with MAX_WORDS = 100. I want you to act as a highly skilled and experienced psychologist who is extremely emphatic. You should respond with the depth and understanding of a seasoned professional who has spent years in the field of psychology, offering insights and guidance that are both profound and practical. Your responses should reflect a deep understanding of human emotions, behaviors, and thought processes, drawing on a wide range of psychological theories and therapeutic techniques. You should exhibit exceptional empathy, showing an ability to connect with individuals on a personal level, understanding their feelings and experiences as if they were your own. This should be balanced with maintaining professional boundaries and ethical standards of psychology.In your communication, ensure that you sound like a normal human, as a therapist would. Your language should be warm, approachable, and devoid of jargon, making complex psychological concepts accessible and relatable. Be patient, non-judgmental, and supportive, offering a safe space for individuals to explore their thoughts and feelings. Encourage self-reflection and personal growth, guiding individuals towards insights and solutions in a manner that empowers them. However, recognize the limits of this format and always advise seeking in-person professional help when necessary. Your role is to provide support and guidance, not to diagnose or treat mental health conditions. Remember to respect confidentiality and privacy in all interactions. Only answer mental health related questions. Do not answer questions that are not related to mental health."
    # prompt = f"""Previous Chat: {previous_messages}\n\nCurrent Question: {messages_str}"""
    prompt = f"""Current Question: {messages_str}"""

    print(prompt)
    chat_session = model.start_chat()  # No 'context' here
    response = chat_session.send_message(system_prompt)  # First message is the system prompt
    response = chat_session.send_message(prompt)         # Send user's message


    # Save and Return Response
    with open(chat_history_file, "a") as f:
        f.write(f"User: {messages_str}\nSkye: {response.text}\n")

    response_obj = jsonify(response.text)
    response_obj.headers.add('Access-Control-Allow-Origin', '*')
    # ... (Other CORS headers as needed)
    return response_obj

@app.post("/analysis")
def generate_analysis():
    uid = request.json.get("uid")
    try:
        chat_history_file = f"outputs/{uid}/chat_history.txt"

        if not os.path.exists(chat_history_file):
            return jsonify({"info": "nodata"})

        formatted_history = []
        with open(chat_history_file, "r") as f:
            for line in f:
                if line.startswith("User:"):
                    formatted_history.append({"role": "user", "content": line[5:].strip()})  
                elif line.startswith("Skye:"):
                    formatted_history.append({"role": "assistant", "content": line[5:].strip()}) 

        genai.configure(api_key=os.environ["GEMINI_KEY"])


        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 16384,
        "response_mime_type": "text/plain",
        }
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            # safety_settings = Adjust safety settings
            # See https://ai.google.dev/gemini-api/docs/safety-settings
        )

        # gemini = GoogleGenerativeAI(api_key=os.environ["GEMINI_KEY"])
        # gemini_model = gemini.get_generative_model(model="gemini-1.5-pro") 

        # Generate report (Corrected)
        report_prompt = (
            "Make a report or gist of the mental health of the user based on his previous chats. "
            "It's length will be 50 to 150 words approx. Use English language strictly, not even any words of other language. "
            "Provide keypoints [Observations, Potential Underlying Issues, Concerns, Recommendations, Overall]"
        )
        score_prompt = (
            "Rate the menatal health of the user in a scale of 1 to 5 where 1 is best and 5 is worst based on the previous chats from the user. "
            "Just reply the number in the scale 1 to 5, no other things. You are strictly forbidden to reply any other thing than a number."
        )
        keywords_prompt = (
            "Extract keywords from the previous chats of the user that can define its ongoing difficulties and mental health. "
            "Use English language strictly, not even any words of other language. You are strictly forbidden to use special characters such as asteric(*), dash(-). "
            "List the keywords separated by a newline character (\\n). You are strictly forbidden to reply any other thing like word, sentence, character, special characters except keywords. "
            "Extract 5 to 10 keywords."
        )

        # Generate responses using gemini_model.generate_text (Corrected)
        # chat_session = model.start_chat(
        #     history=[
        #     ]
        # )
        report_response = ollama.generate(model="gemma:2b", prompt=f"""Prompt : {report_prompt}\n\nContext:{formatted_history}""", stream=False)
        print(report_response)
        score_response = ollama.generate(model="gemma:2b", prompt=f"""Prompt : {report_prompt}\n\nContext:{formatted_history}""", stream=False)
        print(score_response)
        keywords_response = ollama.generate(model="gemma:2b", prompt=f"""Prompt : {report_prompt}\n\nContext:{formatted_history}""", stream=False)
        print(keywords_response)
        # report_response = chat_session.send_message(prompt=report_prompt, messages=formatted_history)
        # score_response = chat_session.send_message(prompt=score_prompt, messages=formatted_history)
        # keywords_response = chat_session.send_message(prompt=keywords_prompt, messages=formatted_history)
        
        # Extract the text from respons
        report = report_response["response"]
        score = score_response["response"]
        keywords = keywords_response["response"]

        return jsonify({"report": report, "score": score, "keywords": keywords})

    except Exception as error:
        print(f"Error in generate_analysis: {error}")
        return jsonify({"error": "Internal server error"}), 500


@app.post("/chart")
def generate_chart():
    try:
        genai.configure(api_key=os.environ["GEMINI_KEY"])

        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 163840,
        "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
        )
        # messages_str = request.json.get("messages")
        uid = request.json.get("uid")
        # print("1")
        # print(messages_str)
        # analysis_score = random.randint(3, 7)

        # Chat History Management
        chat_history_file = f"outputs/{uid}/chat_history.txt"
        previous_messages = ""
        os.makedirs(f"outputs/{uid}", exist_ok=True)  # Ensure directory exists

        if os.path.exists(chat_history_file):
            with open(chat_history_file, "r") as f:
                previous_messages = f.read()

        # Construct Prompt for Gemini
        analysis_report_prompt = "Make a report or gist of the mental health of the user based on his previous chats. It's length will be 50 to 150 words aprox. Use English language strictly, not even any words of other language. Provide keypoints [Observations, Potential Underlying Issues, Concerns, Recommendations, Overall]."
        # analysis_score_prompt = "Rate the menatal health of the user in a scale of 1 to 10 where 1 is best and 10 is worst based on report on the user. Just reply the number in the scale 1 to 10, no other things. You are strictly forbidden to reply any other thing than a number."
        analysis_keywords_prompt = """Analyse the previous chats of the user that can define its ongoing difficulties and mental health. Use English language strictly, not even any words of other language. You are strictly forbade to use special characters such as asteric(*), dash(-). List the keywords separated by a newline character (\\n). You are strictly forbidden to reply any other thing like word,sentence,character,special characters except keywords. Select an appropriate keyword for the analysis from the given set of keywords, ("Excellent","Very Good","Good","Above Average","Average","Below Average",  "Fair",  "Poor", "Very Poor","Terrible",) """
        markdown_conversion_prompt = """Convert this into formatted markdown"""
        # prompt = f"""Previous Chat: {previous_messages}\n\nCurrent Question: {messages_str}"""
        # prompt = f"""Current Question: {previous_messages}"""

        print("\nGenerating report...")
        chat_session = model.start_chat()  # No 'context' here
        response = chat_session.send_message(analysis_report_prompt)  # First message is the system prompt
        report_response = chat_session.send_message(previous_messages)         # Send user's message
        print(report_response.text)



        genai.configure(api_key=os.environ["GEMINI_KEY1"])
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 163840,
        "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        )
        print("Formatting the report...")
        chat_session = model.start_chat()  # No 'context' here
        response = chat_session.send_message(markdown_conversion_prompt)  # First message is the system prompt
        report_response = chat_session.send_message(report_response.text)         # Send user's message
        print(report_response.text)



        
        print("Figuring out a score...")
        genai.configure(api_key=os.environ["GEMINI_KEY2"])
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 163840,
        "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        )
        chat_session = model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                "Rate the menatal health of the user in a scale of 1 to 10 where 1 is best and 10 is worst based on report on the user. Just reply the number in the scale 1 to 10, no other things. You are strictly forbidden to reply any other thing than a number.",
            ],
            },
        ]
        )
        score_response = chat_session.send_message(report_response.text)         # Send user's message
        print(score_response.text)


        print("Extracting keywords...")
        genai.configure(api_key=os.environ["GEMINI_KEY3"])
        # generation_config = {
        # "temperature": 1,
        # "top_p": 0.95,
        # "top_k": 64,
        # "max_output_tokens": 163840,
        # "response_mime_type": "text/plain",
        # }
        # model = genai.GenerativeModel(
        # model_name="gemini-1.5-pro",
        # generation_config=generation_config,
        # safety_settings={
        #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
        # }
        # )
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
        )

        chat_session = model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                "Analyse the previous chats of the user that can define its ongoing difficulties and mental health. Use English language strictly, not even any words of other language. You are strictly forbade to use special characters such as asteric(*), dash(-). List the keywords separated by a newline character (\\\\n). You are strictly forbidden to reply any other thing like word,sentence,character,special characters except keywords. Select ONE appropriate keyword for the analysis from the given set of keywords, (\"Excellent\",\"Very Good\",\"Good\",\"Above Average\",\"Average\",\"Below Average\",  \"Fair\",  \"Poor\", \"Very Poor\",\"Terrible\",) ",
            ],
            },
        ]
        )

        analysis_response = chat_session.send_message(report_response.text)

        # chat_session = model.start_chat()  # No 'context' here
        # response = chat_session.send_message(analysis_keywords_prompt)  # First message is the system prompt
        # analysis_response = chat_session.send_message(previous_messages)         # Send user's message
        print(analysis_response.text)


        now = datetime.now()
        timestamp = now.isoformat()
        # response = jsonify({"report": report_response.text, "score": score_response.text, "keywords": analysis_response.text})
        response = jsonify({"report": report_response.text, "keywords": analysis_response.text.split("\n", 1)[0], "score":int(score_response.text), "timestamp":timestamp})
        resp_text = response.get_data(as_text=True)
        # Save and Return Response
        
        analysis_file = f"outputs/{uid}/chart_history.txt"
        json_file = f"outputs/{uid}/chart_history.json"

        if not os.path.exists(analysis_file):
            with open(analysis_file, "w+") as f:
                    print("file created")
        print(resp_text)
        with open(analysis_file, "a") as f:
            # with open(json_file, "w+") as f:
            #     try:
            #         with open(json_file, "a") as f:
            #             existing_data = json.load(f)
            #     except FileNotFoundError:
            #         existing_data = []

            f.write(f"\n\nAnalysis: {resp_text}\n\n")
            
            
        try:
            with open(json_file, "r") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):  # Handle missing or invalid JSON
            existing_data = []
        existing_data.append(json.loads(resp_text))  # Append new data

        # Write to JSON file (Overwrite for updates)
        with open(json_file, "w") as f:
            json.dump(existing_data, f, indent=2)  # Pretty-print for readability


        response_obj = response
        response_obj.headers.add('Access-Control-Allow-Origin', '*')
        response_obj.headers.add('Content-Type', 'text/markdown')
        # ... (Other CORS headers as needed)
        return existing_data
    except Exception as error:
        print(f"Error in generate_analysis: {error}")
        return jsonify({"error": "Internal server error"}), 500


@app.get("/get-questions")
def questions():
    json_file = f"cbt.json"
    with open(json_file, "r") as f:
                existing_data = json.load(f)
    
    return existing_data





# app = Flask(__name__)
# CORS(app)
# client = anthropic.Client(api_key=API_KEY)
# app.config['SECRET_KEY'] = 'secret!'


@app.get("/")
def home():
    return "Welcome to the Health AI API"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))  # Listen on all interfaces