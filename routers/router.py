from flask import Blueprint, jsonify
from controllers.chat import connect_with_chatbot
from controllers.analysis import do_analysis, get_analysis
from middlewares.genUserId import user_middleware
from controllers.user import (
    signup,
    login,
    is_user,
    logout,
    signup_with_google
)

router = Blueprint("router", __name__)

@router.route("/cron")
def cron_job():
    return jsonify({"message": "hello"}), 200


router.route("/chat", methods=["GET"])(user_middleware(connect_with_chatbot))
router.route("/analysis", methods=["GET"])(user_middleware(do_analysis))
router.route("/fetchanalysis", methods=["GET"])(user_middleware(get_analysis))
router.route("/signup", methods=["POST"])(signup)
router.route("/signupWithGoogle", methods=["POST"])(signup_with_google)
router.route("/login", methods=["POST"])(login)
router.route("/isUser", methods=["GET"])(is_user)
router.route("/logout", methods=["GET"])(logout)