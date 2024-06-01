from firebase import admin

async def decode_auth_token(token: str) -> str | None:
    try:
        id_token = token.split(" ")[1]
        decoded_token = await admin.auth().verify_id_token(id_token)
        email = decoded_token.get("email")
        return email
    except Exception as error:
        print(error)
        return None