from models import db

class ChatHist(db.Document):
    userId = db.StringField(required=True)
    timestamp = db.DateTimeField(default=db.datetime.datetime.now)
    prompt = db.StringField(required=True)
    response = db.StringField(required=True)

    def to_dict(self):
        return {
            "userId": self.userId,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "response": self.response,
        }