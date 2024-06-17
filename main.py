import requests as rq
import logging
from flask import Flask, request, jsonify, Response
import anthropic
import base64
import httpx
# import ollama                                                                                                                                                                               
import random
import uuid
# import openmeteo_requests
import ollama
import requests_cache
import pandas as pd
from retry_requests import retry
from flask import Flask, request, jsonify
from flask_cors import CORS  
import os
from google.generativeai import GenerativeModel
from google.api_core.client_options import ClientOptions
import google.generativeai as genai

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


@app.post("/chat")
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
    system_prompt = """You're skye, a mental health assistant. talk with the person carefully in a happy manner"""
    print(messages)
    response = ollama.generate(model="gemma:2b", prompt=f"""Prompt : {system_prompt}\n\nContext:{messages}""", stream=False)
    
    print(response["response"])
    with open(f"outputs/{uid}/chat_history.txt", "a") as f:
        session_response = f"""User : {messages_str}\n\nSage : {response["response"]}"""
        f.write(session_response)
                
    response_obj = jsonify(response["response"].replace("\n",""))
    response_obj.headers.add('Access-Control-Allow-Origin', '*') # Allow requests from any origin
    return response_obj 


genai.configure(api_key=os.environ["GEMINI_KEY"])

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

@app.post("/chat-gemini")
def chatgemini():
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



# app = Flask(__name__)
# CORS(app)
# client = anthropic.Client(api_key=API_KEY)
# app.config['SECRET_KEY'] = 'secret!'


@app.get("/")
def home():
    return "Welcome to the Health AI API"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))  # Listen on all interfaces