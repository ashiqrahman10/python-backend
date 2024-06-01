import os
from dotenv import load_dotenv
from flask import request, jsonify
import aiohttp
from gemini.chat import start_gemini_chat
from models.ChatHist import ChatHist
from gemini.analysisPrompts import (
    analysis_keywords_prompt,
    analysis_report_prompt,
    analysis_score_prompt,
)
from models.Report import Report
from models.User import User

load_dotenv()

async def do_analysis():
    try:
        user_id = request.userId
        if not user_id:
            return jsonify({"Error": "UserId not found"}), 401

        analysis = await generate_analysis(user_id)

        if analysis.get("info") == "nodata":
            return jsonify({"msg": "nochatdata"}), 200

        report_data = await Report.create(
            {
                "userId": user_id,
                "keywords": analysis.get("keywords"),
                "analysis": analysis.get("report"),
                "score": analysis.get("score"),
            }
        )
        try:
            user = await User.find_one({"id": user_id})
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{os.getenv('EMAIL_API_URL')}/welcomeEmail",
                    json={
                        "emailId": user.email,
                        "score": analysis.get("score"),
                        "analysis": analysis.get("report"),
                        "keywords": analysis.get("keywords"),
                    },
                ) as response:
                    if response.status != 200:
                        print("error sending the message")
        except Exception as error:
            print(error)
            print("error sending the message")
        return (
            jsonify({"data": report_data.to_dict()}),
            200,
        )
    except Exception as error:
        print(error)
        return jsonify({"msg": "Internal Server Error"}), 500


async def generate_analysis(user_id: str) -> dict:
    try:
        if not user_id:
            # through err
            return {}
        found_hist = await ChatHist.find({"userId": user_id}).sort("timestamp", 1)

        if not found_hist:
            return {"info": "nodata"}

        found_hist_for_gemini = []
        for conv in found_hist:
            found_hist_for_gemini.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": conv.prompt,
                        },
                    ],
                }
            )
            found_hist_for_gemini.append(
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": conv.response,
                        },
                    ],
                }
            )

        # generate report
        chat = start_gemini_chat(found_hist_for_gemini)
        result = await chat.send_message(analysis_report_prompt)
        response = await result.response
        report = response.text()

        # generate score
        chat = start_gemini_chat(found_hist_for_gemini)
        result = await chat.send_message(analysis_score_prompt)
        response = await result.response
        score = int(response.text())

        # generate keywords
        chat = start_gemini_chat(found_hist_for_gemini)
        result = await chat.send_message(analysis_keywords_prompt)
        response = await result.response
        keywords_resp = response.text()
        keywords = [
            kw.strip()
            for kw in keywords_resp.replace("[^a-zA-Z0-9 \n]", "").strip().split("\n")
            if kw.strip().lower()
            not in ["keyword", "keywords", ""]
        ]
        # console.log(keywords);

        return {"report": report, "score": score, "keywords": keywords}
    except Exception as error:
        print(error)
        return {}


async def get_analysis():
    try:
        if not request.userId:
            return jsonify({"msg": "UserId not found"}), 401
        user_id = request.userId

        reports = (
            await Report.find({"userId": user_id})
            .sort("timestamp", -1)
            .to_list()
        )

        return jsonify({"data": [report.to_dict() for report in reports]}), 200
    except Exception as error:
        print(error)
        return jsonify({"msg": "Internal Server Error"}), 500