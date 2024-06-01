import os
from uuid import uuid4
from dotenv import load_dotenv
from flask import request, jsonify
import asyncio
from websockets import connect
from urllib.parse import urlencode
from gemini.chat import start_gemini_chat
from models.ChatHist import ChatHist

load_dotenv()


async def connect_with_chatbot():
    try:
        user_id = request.userId
        if not user_id:
            return jsonify({"Error": "UserId not found"}), 401
        found_hist = await ChatHist.find({"userId": user_id}).sort("timestamp", 1)

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

        room_id = str(uuid4())
        params = {"id": room_id, "isServer": True}
        websocketserver_link = (
            f"{os.getenv('WEBSOCKET_SERVER')}?{urlencode(params)}"
        )

        async def websocket_handler(uri):
            async with connect(uri) as websocket:
                await websocket.send(
                    '{"type": "server:connected"}'
                )
                async for message in websocket:
                    try:
                        data = eval(message)
                        if data.get("type") == "client:chathist":
                            await websocket.send(
                                '{"type": "server:chathist", "data": '
                                f"{[hist.to_dict() for hist in found_hist]}}}"
                            )
                        elif data.get("type") == "client:prompt":
                            if not data.get("prompt"):
                                # throw err
                                return

                            # Prompt by the user sent to gemini
                            chat = start_gemini_chat(found_hist_for_gemini)

                            result = await chat.send_message_stream(data.get("prompt"))
                            resp_text = ""
                            await websocket.send('{"type": "server:response:start"}')
                            async for chunk in result.stream:
                                chunk_text = chunk.text()
                                await websocket.send(
                                    '{"type": "server:response:chunk", "chunk": '
                                    f'"{chunk_text}"}}'
                                )
                                resp_text += chunk_text
                            await websocket.send('{"type": "server:response:end"}')
                            # should be stored in the db
                            await ChatHist.create(
                                {
                                    "userId": user_id,
                                    "prompt": data.get("prompt"),
                                    "response": resp_text,
                                }
                            )
                    except Exception as error:
                        print(error)
        asyncio.create_task(websocket_handler(websocketserver_link))
        return jsonify({"chatId": room_id}), 200
    except Exception as error:
        print(error)
        return jsonify({"message": "Websocket connection error"}), 404