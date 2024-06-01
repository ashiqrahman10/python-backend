from models import db

class Report(db.Document):
    userId = db.StringField(required=True)
    score = db.IntField()
    keywords = db.ListField(db.StringField())
    analysis = db.StringField()
    timestamp = db.DateTimeField(default=db.datetime.datetime.now)

    def to_dict(self):
        return {
            "userId": self.userId,
            "score": self.score,
            "keywords": self.keywords,
            "analysis": self.analysis,
            "timestamp": self.timestamp,
        }