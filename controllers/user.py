from flask import request, jsonify, make_response
from uuid import uuid4
from models.User import User
from models.Report import Report
from controllers.analysis import generate_analysis
from firebase.auth import decode_auth_token

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