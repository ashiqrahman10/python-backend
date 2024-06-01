from models import db

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