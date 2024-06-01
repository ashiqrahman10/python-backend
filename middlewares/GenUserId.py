from uuid import uuid4
from functools import wraps
from flask import request, make_response

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