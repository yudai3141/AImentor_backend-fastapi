from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired
import uuid
import motor.motor_asyncio
import json

from app.routers import chat2, weekly_chat
from dotenv import load_dotenv
load_dotenv()

MONGO_URL = os.getenv("MONGOURL")
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
NODEJS_URL = os.getenv("NODEJS_URL", "http://localhost:5002")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        NODEJS_URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------
# GPT用: セッション保存 (sessionsコレクション使用) 
# -----------------------------------------
class MongoDBSessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str, mongo_url: str, max_age: int = 3600):
        super().__init__(app)
        self.signer = TimestampSigner(secret_key)
        self.max_age = max_age
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
        self.db = self.client['session_db']
        self.collection = None  # コレクションは動的に設定

    async def dispatch(self, request: StarletteRequest, call_next):
        # GPT用のエンドポイントのみ処理
        if not request.url.path.startswith('/api/gpt'):
            return await call_next(request)

        # URLのパスからユーザーIDを取得
        path_parts = request.url.path.split('/')
        user_id = 'default'
        for i, part in enumerate(path_parts):
            # /api/gpt/api/gpt/{user_id} の形式をチェック
            print("Checking path part:", part, "at index", i)
            if part == 'gpt' and len(path_parts) > i + 1:
                user_id = path_parts[-1]  # 最後の要素がuser_id
                print("Found user_id:", user_id)
        
        # ユーザーIDに基づいてコレクションを設定
        self.collection = self.db[f'sessions_{user_id}']

        session_id = request.cookies.get('fastapi_session_id')
        request.state.session = {}

        if session_id:
            try:
                session_id = self.signer.unsign(session_id, max_age=self.max_age).decode()
                session_data = await self.collection.find_one({'_id': session_id})
                if session_data:
                    request.state.session = session_data['data']
            except (BadSignature, SignatureExpired):
                pass

        response = await call_next(request)

        new_session_id = str(uuid.uuid4())
        signed_session_id = self.signer.sign(new_session_id).decode()
        await self.collection.update_one(
            {'_id': new_session_id}, 
            {'$set': {'data': request.state.session}}, 
            upsert=True
        )
        response.set_cookie(
            'fastapi_session_id', 
            signed_session_id, 
            max_age=self.max_age, 
            httponly=True
        )
        return response


# -----------------------------------------
# 週次用: セッション保存 (goal_numに応じてコレクションを変える) 
# -----------------------------------------
class MongoDBWeeklySessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str, mongo_url: str, max_age: int = 3600):
        super().__init__(app)
        self.signer = TimestampSigner(secret_key)
        self.max_age = max_age
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
        self.db = self.client['session_db']
        self.collection = None

    async def dispatch(self, request: StarletteRequest, call_next):
        if not request.url.path.startswith('/api/weekly-chat'):
            return await call_next(request)

        # URLのパスから目標番号を取得
        path_parts = request.url.path.split('/')
        goal_num = 0
        user_id = 'default'
        for i, part in enumerate(path_parts):
            if part == 'weekly-chat' and i + 1 < len(path_parts):
                try:
                    goal_num = int(path_parts[i + 1])
                    if i + 2 < len(path_parts):
                        user_id = path_parts[i + 2]
                except (ValueError, IndexError):
                    pass
        
        print(f"リクエストから取得したgoal_num: {goal_num}")

        session_id = request.cookies.get('weekly_session_id')
        request.state.session = {}

        print("session_idの確認")
        if session_id:
            print(f"session_id: {session_id}")
            try:
                decoded_id = self.signer.unsign(session_id, max_age=self.max_age).decode()
                print(f"goal_num: {goal_num}")
            except (BadSignature, SignatureExpired, ValueError, IndexError):
                pass

        self.collection = self.db[f'weekly_sessions_{goal_num + 1}_{user_id}']

        if session_id:
            try:
                session_id = self.signer.unsign(session_id, max_age=self.max_age).decode()
                session_data = await self.collection.find_one({'_id': session_id})
                if session_data:
                    request.state.session = session_data['data']
            except (BadSignature, SignatureExpired):
                pass

        response = await call_next(request)

        # goal_numを含むセッションIDを生成
        new_uuid = str(uuid.uuid4())
        new_session_id = f"{goal_num}_{new_uuid}"
        signed_session_id = self.signer.sign(new_session_id).decode()
        await self.collection.update_one(
            {'_id': new_session_id}, 
            {'$set': {'data': request.state.session}}, 
            upsert=True
        )
        response.set_cookie(
            'weekly_session_id', 
            signed_session_id, 
            max_age=self.max_age, 
            httponly=True
        )
        return response


# ミドルウェアを追加
app.add_middleware(MongoDBSessionMiddleware, secret_key=SECRET_KEY, mongo_url=MONGO_URL, max_age=3600)
app.add_middleware(MongoDBWeeklySessionMiddleware, secret_key=SECRET_KEY, mongo_url=MONGO_URL, max_age=3600)

# ルーターの登録
app.include_router(chat2.router, prefix="/api/gpt", tags=["gpt"])
app.include_router(weekly_chat.router, prefix="/api/weekly-chat", tags=["weekly"])

@app.get("/")
async def root():
    return {"message": "AI Chat API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)