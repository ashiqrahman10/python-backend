import os
import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials

load_dotenv()
cred = credentials.Certificate(eval(os.getenv("FIREBASE_KEY")))

admin = firebase_admin.initialize_app(cred)