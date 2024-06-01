import os
from dotenv import load_dotenv
from models import db

load_dotenv()


async def connect_db():
    db.connect(host=os.getenv("MONGO_URI"))