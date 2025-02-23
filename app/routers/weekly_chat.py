from fastapi import FastAPI, Request, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired
import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId
import uuid
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from openai import AsyncOpenAI  # ここはOK（公式openaiライブラリの非同期版）
import json
import sys
from enum import Enum
from datetime import datetime, timedelta

# 環境変数のロード
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_url = os.getenv("MONGOURL")
secret_key = os.getenv("SECRET_KEY", "your_secret_key")

# OpenAIクライアントの初期化
client = AsyncOpenAI(api_key=openai_api_key)

router = APIRouter()

class WeeklyMeetingStage:
    ACHIEVEMENT_CHECK = 1
    CURRENT_SITUATION = 2
    NEXT_GOAL = 3

class ChatMessage(BaseModel):
    role: str
    content: str

class TaskStatus(str, Enum):
    NOT_STARTED = 'not_started'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'

class TaskProgress(BaseModel):
    title: str
    completed: bool
    importance: int
    status: TaskStatus = TaskStatus.NOT_STARTED

class NumericalProgress(BaseModel):
    current_achievement_num: int

class TFProgress(BaseModel):
    task_status: List[bool]  # タスクの順序通りの完了状態リスト

# ShortTermGoalProgressを他のモデルより先に定義
class ShortTermGoalProgress(BaseModel):
    shortTerm_goal: str
    KPI: str
    numerical_or_TF: str
    KPI_indicator: Optional[int] = None
    current_achievement_num: Optional[int] = None
    weekly_goal_num: Optional[int] = None  # 週ごとの目標値
    importance: Optional[int] = None
    status: TaskStatus = TaskStatus.NOT_STARTED
    tasks: Optional[List[TaskProgress]] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    goal_num: int
    shortTermGoal: Optional[ShortTermGoalProgress] = None
    user_id: Optional[str] = None
    success_experiences: Optional[List[str]] = []
    failure_experiences: Optional[List[str]] = []
    high_level_insights: Optional[List[str]] = []

class InsightAnalysis(BaseModel):
    success_experiences: str
    failure_experiences: str
    high_level_insights: str

class MemoryManager:
    def __init__(self, client: AsyncIOMotorClient):
        self.db = client['your_database_name']
        self.users = self.db['users']
        self.mtg_count = 4  # 分析を行うMTGの回数
    
    async def add_conversation(self, user_id: str, mtg_data: dict):
        """1回分のMTG全体を記録"""
        mtg_summary = {
            "timestamp": datetime.now(),
            "goal": mtg_data["goal"],
            "messages": mtg_data["messages"]  # セッションに保存された全会話履歴
        }
        
        await self.users.update_one(
            {"_id": user_id},
            {
                "$push": {
                    "conversation_history": mtg_summary
                }
            }
        )
        
        # 分析のタイミングをチェック
        await self.check_and_analyze(user_id)
    
    async def check_and_analyze(self, user_id: str):
        """一定回数のMTG後に分析を実行"""
        user_data = await self.users.find_one({"_id": user_id})
        conversations = user_data.get("conversation_history", [])
        
        if len(conversations) >= self.mtg_count:
            # 分析を実行
            insights = await self.analyze_conversations(conversations)
            
            # 永続的な記憶と洞察を更新
            await self.update_user_memory(user_id, insights)
            
            # 分析済みの会話履歴をクリア
            await self.users.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "conversation_history": [],
                        "last_analyzed": datetime.now()
                    }
                }
            )
    
    async def analyze_conversations(self, conversations: List[dict]) -> dict:
        """会話履歴から重要な洞察を生成"""
        prompt = f"""
以下のMTG履歴から、ユーザーに関する重要な洞察を抽出してください：

会話履歴：
{conversations}

以下の3つの観点から分析してください：

1. 成功体験:
- 具体的な成功事例
- 成功要因の分析
- 活用可能な学び

2. 失敗体験:
- 具体的な失敗事例
- 失敗から得られた教訓
- 今後の改善点

3. 高次の洞察:
- 学習パターンの特徴
- 成長を促進する要因
- 潜在的な強み
- 克服すべき課題
"""
        # 例外はすべてExceptionで捕捉
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは優秀な人材開発アナリストです"},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                functions=[{
                    "name": "analyze_insights",
                    "parameters": InsightAnalysis.schema()
                }]
            )
            return InsightAnalysis.parse_raw(response.choices[0].message.content)
        except Exception as e:
            print(f"[analyze_conversations Error] {str(e)}")
            raise

    async def update_user_memory(self, user_id: str, insights: dict):
        """ユーザーの記憶を更新"""
        update_ops = {
            "$addToSet": {
                "success_experiences": {
                    "$each": insights["success_experiences"]
                },
                "failure_experiences": {
                    "$each": insights["failure_experiences"]
                }
            },
            "$set": {
                "high_level_insights": insights["high_level_insights"]
            }
        }
        try:
            await self.users.update_one({"_id": user_id}, update_ops)
        except Exception as e:
            print(f"[update_user_memory Error] {str(e)}")
            raise

class ConversationRequest(BaseModel):
    conversations: List[dict]


@router.post("/analyze-memory")
async def analyze_memory(request: Request):
    """会話履歴から重要な洞察を生成するAPIエンドポイント"""
    try:
        data = await request.json()
        raw_messages = data.get("conversations", [{}])[0].get("messages", [])
        
        # メッセージを整形（roleとcontentのみ抽出）
        formatted_messages = []
        for msg in raw_messages:
            formatted_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })
        
        print("\n=== 1. 整形後の会話履歴 ===")
        print(json.dumps(formatted_messages, indent=2, ensure_ascii=False))
        
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたは優秀な人材開発アナリストです。会話履歴から具体的な洞察を抽出し、文字列として出力してください。"},
                {"role": "user", "content": f"""
以下のMTG履歴から、ユーザーに関する重要な洞察を抽出してください：

会話履歴：
{formatted_messages}

以下の3つの観点から分析してください：

1. 成功体験: 具体的な成功事例、成功要因、学びを文章で説明
2. 失敗体験: 具体的な失敗事例、教訓、改善点を文章で説明
3. 高次の洞察: 学習パターン、成長要因、強み、課題を文章で説明

***もし該当する内容がない場合は、その項目に"現状無し"を設定してください。
また、内容は、該当会話を知らない人が見てもわかるように記述してください。
"""}
            ],
            response_format=InsightAnalysis
        )
        
        response_data = completion.choices[0].message.parsed.dict()
        # print("\n=== 2. GPTへのプロンプト（生データ） ===")
        # print(completion)
        # print("\n=== 2. GPTへのプロンプト ===")
        # print(completion.choices[0].message.content)
        # print("\n=== 3. AIからの応答（解析前） ===")
        # print(json.dumps(response_data, indent=2, ensure_ascii=False))
        
        return response_data
    except Exception as e:
        print("\n=== エラー詳細 ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/weekly-chat/{goal_num}/{user_id}")
async def handle_weekly_meeting(goal_num: int, user_id: str, request: Request, data: ChatRequest):
    try:
        print(f"data.success_experiences: {data.success_experiences}")
        session = request.state.session
        user_id = data.user_id
        
        current_goal = data.shortTermGoal
        print(f"current_goal: {current_goal}")
        print(f"KPI: {current_goal.KPI}")

        if session.get("messages"):        # メッセージの処理
            user_messages = [{"role": msg.role, "content": msg.content} for msg in data.messages]
            session_messages = convert_to_langchain_messages(session["messages"])
            user_langchain_messages = convert_to_langchain_messages(user_messages)

            chat_messages = session_messages + user_langchain_messages
        
        # セッションの初期化
        else:
            session.setdefault("messages", [])
            session.setdefault("stage", WeeklyMeetingStage.ACHIEVEMENT_CHECK)
            session.setdefault("goal_num", 0)
            session.setdefault("shortTermGoal", {})  # 配列ではなく辞書として初期化
            session.setdefault("isTerminated", False)
            session.setdefault("is_confirmed", False)
            
            # 最初のシステムメッセージを設定
            system_message = weekly_first_system_message_content(current_goal)
            chat_messages = [SystemMessage(content=system_message)]
            session_messages = chat_messages  # セッションにシステムメッセージを保存
            session["messages"] = [message_to_dict(m) for m in session_messages]  # 辞書形式で保存
            user_langchain_messages = []
    
        chat = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # アクション決めのステージでフィードバックを行う
        if session["stage"] == WeeklyMeetingStage.NEXT_GOAL:
            
            # フィードバック付きの応答を生成
            gpt_message = await generate_response(
                chat_messages, 
                data.success_experiences,  # dataから参照
                data.failure_experiences,  # dataから参照
                data.high_level_insights   # dataから参照
            )
        else:
            # 通常の応答を生成
            response = chat(chat_messages)
            gpt_message = response.content

        session["is_confirmed"] = False # ユーザーの確認を受けていない場合はFalse   

        if "達成率の確認が完了しました" in gpt_message:
            session["is_confirmed"] = True
            session["stage"] = WeeklyMeetingStage.CURRENT_SITUATION
            gpt_message += "次に、今週の状況について伺います。今週の予定や割り当て可能な時間について教えてください"
            
            # メッセージの追加方法を修正
            session_messages.extend(user_langchain_messages)
            session_messages.append(AIMessage(content=gpt_message))
            session_messages.append(SystemMessage(content=weekly_third_system_message_content()))
            session["messages"] = [message_to_dict(m) for m in session_messages]

            if current_goal.numerical_or_TF == "TF":
                completion = await client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "あなたの仕事は、ユーザの小目標の更新です"},
                        {"role": "user", "content": f"""
    現在の小目標の状況：{current_goal}
    会話ログ：{[msg.content if hasattr(msg, 'content') else msg['content'] for msg in chat_messages]}
    最新の応答：{gpt_message}

    上記の会話履歴から、小目標とそのタスクについて、status：現在の状況を更新してください。その他の項目は変更しないでください。
    """},
                    ],
                    response_format=ShortTermGoalProgress,
                )

            else: # numericalの場合 
                completion = await client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "あなたの仕事は、ユーザの小目標の更新です"},
                        {"role": "user", "content": f"""
現在の小目標の状況：{current_goal}
会話ログ：{[msg.content if hasattr(msg, 'content') else msg['content'] for msg in chat_messages]}
最新の応答：{gpt_message}

会話の流れから、current_achievement_num：現在の達成値、およびstatus：現在の状況を更新してください。その他の項目は変更しないでください。
    """},
                    ],
                    response_format=ShortTermGoalProgress,
                )
            
            parsed_data = completion.choices[0].message.parsed.dict()
            print(f"parsed_data: {parsed_data}")
            # 現在のインデックスの小目標を更新
            session["shortTermGoal"] = {
                "shortTerm_goal": parsed_data["shortTerm_goal"],
                "KPI": parsed_data["KPI"],
                "numerical_or_TF": parsed_data["numerical_or_TF"],
                "KPI_indicator": parsed_data["KPI_indicator"],
                "current_achievement_num": parsed_data["current_achievement_num"],
                "weekly_goal_num": parsed_data["weekly_goal_num"],
                "importance": parsed_data["importance"],
                "status": parsed_data["status"],
                "tasks": parsed_data["tasks"]
            }

        if "現状の理解が完了しました" in gpt_message:
            session["stage"] = WeeklyMeetingStage.NEXT_GOAL
            if current_goal.numerical_or_TF == "TF":
                gpt_message += "次に、今週取り組むアクションを決めます。どのアクションに取り組みましょうか"
                
                # メッセージの追加方法を修正
                session_messages.extend(user_langchain_messages)
                session_messages.append(AIMessage(content=gpt_message))
                session_messages.append(SystemMessage(content=weekly_forth_system_message_content(current_goal)))
            else:
                gpt_message += "次に、今週の目標数値を決めます。今週の目標数値を考えましょう。"
                
                # メッセージの追加方法を修正
                session_messages.extend(user_langchain_messages)
                session_messages.append(AIMessage(content=gpt_message))
                session_messages.append(SystemMessage(content=weekly_fifth_system_message_content(current_goal)))
            
            session["messages"] = [message_to_dict(m) for m in session_messages]

        if "週次コーチングが完了しました" in gpt_message:
            if current_goal.numerical_or_TF == "TF":
                completion = await client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "あなたの仕事は、ユーザの小目標の更新です"},
                        {"role": "user", "content": f"""
    現在の小目標の状況：{current_goal}
    会話ログ：{[msg.content if hasattr(msg, 'content') else msg['content'] for msg in chat_messages]}
    最新の応答：{gpt_message}

    上記の会話履歴から、小目標とそのタスクについて、status：着手状況を更新してください。その他の項目は変更しないでください。
    """},
                    ],
                    response_format=ShortTermGoalProgress,
                )

            else: # numericalの場合 
                completion = await client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "あなたの仕事は、ユーザの小目標の更新です"},
                        {"role": "user", "content": f"""
現在の小目標の状況：{current_goal}
会話ログ：{[msg.content if hasattr(msg, 'content') else msg['content'] for msg in chat_messages]}
最新の応答：{gpt_message}

会話の流れから、weekly_goal_num：今週の目標値、current_achievement_num：現状の達成値、およびstatus：現在の状況を更新してください。その他の項目は変更しないでください。
    """},
                    ],
                    response_format=ShortTermGoalProgress,
                )
            
            parsed_data = completion.choices[0].message.parsed.dict()
            print(f"parsed_data: {parsed_data}")
            # 現在のインデックスの小目標を更新
            session["shortTermGoal"] = {
                "shortTerm_goal": parsed_data["shortTerm_goal"],
                "KPI": parsed_data["KPI"],
                "numerical_or_TF": parsed_data["numerical_or_TF"],
                "KPI_indicator": parsed_data["KPI_indicator"],
                "current_achievement_num": parsed_data["current_achievement_num"],
                "weekly_goal_num": parsed_data["weekly_goal_num"],
                "importance": parsed_data["importance"],
                "status": parsed_data["status"],
                "tasks": parsed_data["tasks"]
            }

            session["isTerminated"] = True # １つの小目標のコーチングが完了したら、終了フラグを立てる
            session["goal_num"] += 1 # 小目標のインデックスを更新

        else:
            session_messages.extend(user_langchain_messages)
            session_messages.append(AIMessage(content=gpt_message))
            session["messages"] = [message_to_dict(m) for m in session_messages]

        # セッションデータを保存
        request.state.session = session

        return {
            "response": gpt_message,
            "stage": session["stage"],
            "goal_num": session["goal_num"],
            "isTerminated": session["isTerminated"],
            "shortTermGoal": session["shortTermGoal"],
            "is_confirmed": session["is_confirmed"],
        }

    except Exception as e:
        error_message = f'Error in weekly meeting: {str(e)}'
        print(json.dumps({'error': error_message}), file=sys.stderr)
        return JSONResponse({"error": error_message}, status_code=500)

def weekly_first_system_message_content(shortTerm_goal):
    if shortTerm_goal.numerical_or_TF=="TF":
        return f"""
小目標： {shortTerm_goal.shortTerm_goal}

目標アクション：
    {shortTerm_goal.tasks}
あなたは、対話型の会社の上司の役割を持つアシスタントです。
次に、上記の小目標と目標アクションに対して、達成率について補佐をしてください。
コーチングでは以下の事柄を順守してください:
・目標アクションを細分化しない
・肯定的な相槌を交える。ではを使わない
・一度に一つだけ質問を行う
・回答を深掘りする
・何を行ったか・行えば良かったかについてのみ問う
・**1回の発話は長くなりすぎないようにする
以下はコーチングの質問例です：
・どの行動が良かったから達成できたと思うか
・仮に何を行ったら達成できたと思うか
プロセスはユーザとの対話を通じて以下の手順で行います：
1. このステップの目的を話す
2. 各アクションを達成したか否かを決める
3. 短い例示を交えながら、目標アクションのコーチングを行う。ユーザーが十分な理解が得られるまで繰り返す
4. OKであれば現状と解決策をまとめユーザーに確認し、問題があればその理由を考慮して上記のプロセスに戻る。
**ユーザと対話しながらプロセスを着実に実行していき、すべてのプロセスが終了後もしくはユーザが'以上です'といった場合、'達成率の確認が完了しました'と出力してください。"""
    
    else:
        return f"""
小目標： {shortTerm_goal.shortTerm_goal}
KPI: {shortTerm_goal.KPI}
目標数値: {shortTerm_goal.KPI_indicator}

あなたは、対話型の会社の上司の役割を持つアシスタントです。
次に、上記の小目標と目標数値に対して、達成率について補佐をしてください。
コーチングでは以下の事柄を順守してください:
・目標数値を言い換える
・肯定的な相槌を交える。「では」を使わない
・一度に一つだけ質問を行う
・回答を深掘りする
・何を行ったか・行えば良かったかについてのみ問う
・**1回の発話は長くなりすぎないようにする
以下はコーチングの質問例です：
・どの行動が良かったから達成できたと思うか
・仮に何を行ったら達成できたと思うか
プロセスは以下の手順で行います：
1. このステップの目的を話す
2. 進捗を尋ねた後、目標数値を達成したか否か決める
3. 短い例示を交えながら、目標数値のコーチングを行う。ユーザーが十分な理解が得られるまで繰り返す
4. OKであれば現状と解決策をまとめユーザーに確認し、問題があればその理由を考慮して上記のプロセスに戻る。
**ユーザと対話しながらプロセスを着実に実行していき、すべてのプロセスが終了後、もしくはユーザが'以上です'といった場合、'達成率の確認が完了しました'と出力してください。"""

def weekly_third_system_message_content():
    return f"""
あなたは、会社の上司の役割を持つアシスタントです。
次に、ユーザーが何も言語化していない前提で、今週実行するアクション決めのための現状の理解を促すコーチングを実行してください。
現状の理解とは、今週特有の用事や事情などを自覚することです。
コーチングでは以下の事柄を順守してください:
・相槌を交える。ではを使わない
・一度に一つの質問を行う
・原因や理由について問わない
・はい・いいえで回答できない具体的かつポジティブな質問をする
・回答を深掘りする
・アクション決めを行わない
以下は現状の理解を促すコーチングの一例です：
・時間
・体調
プロセスは以下の手順で行います：
1. 話題が変わったことと、このステップの目的を話す
2. 短い例示を交えながら、現状の理解を促すコーチングを行う。ユーザーが十分な理解が得られるまで繰り返す
3. OKであれば現状をまとめた後に終了し、今週全体で使える時間を出力したあと、問題があればその理由を考慮して上記のプロセスに戻る。
**ユーザと対話しながらプロセスを着実に実行していき、すべてのプロセスが終了後、もしくはユーザが'以上です'といった場合、'現状の理解が完了しました'と出力してください。"""

def weekly_forth_system_message_content(shortTerm_goal):
    return f"""
小目標： {shortTerm_goal.shortTerm_goal}
未完了アクション：{shortTerm_goal.tasks}

あなたは、会社の上司の役割を持つアシスタントです。
次に、上記の小目標で、今週実行するアクションを決める補佐をしてください。

ユーザーの過去の経験から以下を考慮して、ユーザに寄り添った方向性を示してください：
#成功体験：過去に効果的だった進め方や達成ペース
・ユーザーは、オンライン講座を参考にすることで、初めてのAPI開発で2つのAPIを成功裏に作成しました。この経験から学んだことは、初心者であっても適切なリソースを利用することで実用的な成果を上げられることです。サポートツールをうまく活用することで、自己能力を超えた成果を達成することができました。
・ユーザーはオンライン講座を活用して、初心者ながらも2つのAPIを作成することに成功しています。これにより、体系的に知識を身につけることができたと述べており、この学習スタイルが有効であったと考えられます。
#失敗体験：過去に避けるべきだった目標設定の問題点
・具体的な失敗事例は示されていませんが、ユーザーはまだAPIをコピーする段階にいることから、オリジナリティの欠如が課題である可能性があります。改善点として、コピーに頼るだけでなく、自分自身の変化や工夫を加えていく必要があると述べています。
・会話には具体的な失敗経験は述べられていませんが、ユーザーはまだ決まっていないAPI内容の変更に関する不確実性があるように見られます。このことから新しい挑戦において具体的な計画を立てないことが教訓として挙げられます。今後は計画を具体化することで未確定要素を減らすことが重要です。
#高次の洞察：ユーザーの実績パターンや現実的な目標設定の傾向
・ユーザーは、オンライン講座を参考にすることで、初めてのAPI開発で2つのAPIを成功裏に作成しました。この経験から学んだことは、初心者であっても適切なリソースを利用することで実用的な成果を上げられることです。サポートツールをうまく活用することで、自己能力を超えた成果を達成することができました。
・ユーザーはプロジェクトの一環として4つのAPIを作成する目標を掲げていました。すでに2つのAPIを完成させており、特にデータベース（DB）との接続を上手く実行したことが成功の要因として挙げられています。これにより、目標の半分を達成しており、技術的なスキルの向上が見られます。

コーチングでは以下の事柄を順守してください:
・相槌を交える。ではを使わない
・一度に一つの質問を行う
・原因や理由について問わない
・アクションを細分化しない

プロセスは以下の手順で行います：
1. 短い例示を交えながら、次のアクション決めのコーチングを行う。ユーザーが十分な理解が得られるまで繰り返す。
2. アクションに向けての具体的な方針決めを行う。
3. OKであれば現状をまとめた後に終了し、問題があればその理由を考慮して上記のプロセスに戻る。

**すべてのプロセスが終了後、'週次コーチングが完了しました'と出力してください。"""
def weekly_fifth_system_message_content(shortTerm_goal):
    return f"""
小目標： {shortTerm_goal.shortTerm_goal}
KPI: {shortTerm_goal.KPI}
目標数値: {shortTerm_goal.KPI_indicator}
進捗: {shortTerm_goal.current_achievement_num}

あなたは、会社の上司の役割を持つアシスタントです。
次に、上記の小目標に対して、今週からの目標数値を決める補佐をしてください。

ユーザーの過去の経験から以下を考慮して、ユーザに寄り添った方向性を示してください：
#成功体験：過去に効果的だった進め方や達成ペース
・ユーザーは、オンライン講座を参考にすることで、初めてのAPI開発で2つのAPIを成功裏に作成しました。この経験から学んだことは、初心者であっても適切なリソースを利用することで実用的な成果を上げられることです。サポートツールをうまく活用することで、自己能力を超えた成果を達成することができました。
・ユーザーはオンライン講座を活用して、初心者ながらも2つのAPIを作成することに成功しています。これにより、体系的に知識を身につけることができたと述べており、この学習スタイルが有効であったと考えられます。
#失敗体験：過去に避けるべきだった目標設定の問題点
・具体的な失敗事例は示されていませんが、ユーザーはまだAPIをコピーする段階にいることから、オリジナリティの欠如が課題である可能性があります。改善点として、コピーに頼るだけでなく、自分自身の変化や工夫を加えていく必要があると述べています。
・会話には具体的な失敗経験は述べられていませんが、ユーザーはまだ決まっていないAPI内容の変更に関する不確実性があるように見られます。このことから新しい挑戦において具体的な計画を立てないことが教訓として挙げられます。今後は計画を具体化することで未確定要素を減らすことが重要です。
#高次の洞察：ユーザーの実績パターンや現実的な目標設定の傾向
・ユーザーは、オンライン講座を参考にすることで、初めてのAPI開発で2つのAPIを成功裏に作成しました。この経験から学んだことは、初心者であっても適切なリソースを利用することで実用的な成果を上げられることです。サポートツールをうまく活用することで、自己能力を超えた成果を達成することができました。
・ユーザーはプロジェクトの一環として4つのAPIを作成する目標を掲げていました。すでに2つのAPIを完成させており、特にデータベース（DB）との接続を上手く実行したことが成功の要因として挙げられています。これにより、目標の半分を達成しており、技術的なスキルの向上が見られます。

コーチングでは以下の事柄を順守してください:
・相槌を交える。ではを使わない
・一度に一つの質問を行う
・原因や理由・手段について問わない

プロセスは以下の手順で行います：
1. 短い例示を交えながら、目標数値決めのコーチングを行う。ユーザーが十分な理解が得られるまで繰り返す。
2. 目標数値を達成するために,具体的に行う内容を決定する。
3. OKであれば現状をまとめた後に終了し、問題があればその理由を考慮して上記のプロセスに戻る。

**すべてのプロセスが終了後、'週次コーチングが完了しました'と出力してください。"""

def convert_to_langchain_messages(messages: List[dict]) -> List[BaseMessage]:
    """
    通常のメッセージ形式からLangChainのメッセージ形式に変換する
    """
    langchain_messages = []
    for message in messages:
        if message["role"] == "system":
            langchain_messages.append(SystemMessage(content=message["content"]))
        elif message["role"] == "user":
            langchain_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            langchain_messages.append(AIMessage(content=message["content"]))
    return langchain_messages

def message_to_dict(message: BaseMessage) -> dict:
    """
    LangChainのメッセージ形式を通常のdict形式に変換する
    """
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

async def extract_progress_from_response(gpt_message: str, current_goal: dict) -> dict:
    """GPTの応答から進捗データを抽出"""
    try:
        if current_goal["numerical_or_TF"] == "TF":
            completion = client.beta.chat.completions.parse(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたの仕事は、文章の要約です"},
                    {"role": "user", "content": f"""
{gpt_message}
上記の会話から、各タスクの完了状態をtrue/falseのリストとして抽出してください。
タスクの順序は以下の通りです：
{[task["title"] for task in current_goal["tasks"]]}
"""}
                ],
                response_format=TFProgress
            )
            
            # タスクの完了状態を更新用のデータ形式に変換
            progress_data = {
                "numerical_or_TF": "TF",
                "tasks": [
                    {
                        "title": task["title"],
                        "completed": status,
                        "importance": task["importance"]
                    }
                    for task, status in zip(current_goal["tasks"], completion.choices[0].message.parsed.task_status)
                ]
            }
        else:
            completion = client.beta.chat.completions.parse(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたの仕事は、文章の要約です"},
                    {"role": "user", "content": f"""
{gpt_message}
上記の会話から、達成した数値を抽出してください。
"""}
                ],
                response_format=NumericalProgress
            )
            
            progress_data = {
                "numerical_or_TF": "numerical",
                "current_achievement_num": completion.choices[0].message.parsed.current_achievement_num
            }

        return progress_data

    except Exception as e:
        print(f"Error extracting progress: {str(e)}")
        return None

async def update_progress(user_id: str, goal_index: int, progress_data: dict):
    """ユーザーの進捗データを更新"""
    client = AsyncIOMotorClient(mongo_url)
    db = client['your_database_name']
    users_collection = db['users']

    try:
        if progress_data["numerical_or_TF"] == "TF":
            # TF目標の場合、完了したタスクを更新
            update_query = {
                f"shortTerm_goals.{goal_index}.tasks.$[task].completed": True
            }
            array_filters = [{"task.title": {"$in": progress_data.get("completed_tasks", [])}}]
            
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_query},
                array_filters=array_filters
            )
        else:
            # numerical目標の場合、達成数を更新
            update_query = {
                f"shortTerm_goals.{goal_index}.current_achievement_num": progress_data.get("current_achievement_num", 0)
            }
            
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_query}
            )

        return True
    except Exception as e:
        print(f"Error updating progress: {str(e)}")
        return False

async def generate_response(messages, success_exp, failure_exp, insights):
    try:
        # 1. 通常の応答を生成
        formatted_messages = []
        for m in messages:
            # 'ai' -> 'assistant', 'human' -> 'user' に変換
            role = "assistant" if m.type == "ai" else "user" if m.type == "human" else m.type
            formatted_messages.append({"role": role, "content": m.content})

        initial_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=formatted_messages
        )
        proposed_action = initial_response.choices[0].message.content

        # 2. フィードバックの生成
        feedback_prompt = f"""
過去の経験を基に、提案された内容を評価してください：

提案内容：
{proposed_action}

ユーザーの経験：
失敗体験（今後の教訓）：{failure_exp}
高次の洞察（暗黙知）：{insights}

失敗体験（今後の教訓）を参照して、提案内容がユーザにとって適切であるかどうか判断し、適切でなければ改善案を生成してください。
問題がない場合は、"OK"と返してください。
"""
        feedback = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": feedback_prompt}]
        )
        feedback_result = feedback.choices[0].message.content
        print(f"feedback_result: {feedback_result}")

        # 3. フィードバックに基づいて応答を生成
        if feedback_result != "OK":
            # 同じ変換ロジックを使用
            formatted_messages = []
            for m in messages:
                role = "assistant" if m.type == "ai" else "user" if m.type == "human" else m.type
                formatted_messages.append({"role": role, "content": m.content})
            
            formatted_messages.append({"role": "assistant", "content": proposed_action})
            formatted_messages.append({"role": "user", "content": f"""
あなたのユーザに対する応答：{proposed_action}
フィードバック：{feedback_result}
改善した形で再度、**ユーザに対しての応答**を再度生成してください。"""})

            final_response = await client.chat.completions.create(
                model="gpt-4o",
                messages=formatted_messages
            )
            final_content = final_response.choices[0].message.content
        else:
            final_content = proposed_action

        print(f"final_content: {final_content}")
        return final_content

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise

@router.post("/generate-advice")
async def generate_advice(request: Request):
    try:
        data = await request.json()
        print("Received data:", data)  # リクエストデータの確認
        chat_history = data.get("messages", [])
        goal = data.get("goal", {})


        advice_prompt = f"""
あなたは優秀なビジネスコーチです。
以下の会話履歴を分析し、次週に向けた具体的なアドバイスを提供してください：

目標：{goal.get('shortTerm_goal')}
KPI：{goal.get('KPI')}

アドバイスは以下の点を含めてください：
1. 今週の取り組みの良かった点
2. 改善できる点
3. 次週に向けた具体的な行動提案
4. モチベーション維持のためのヒント

200文字程度で簡潔にまとめてください。
"""
        print("Generated prompt:", advice_prompt)  # プロンプトの確認
        
        # アドバイス生成用のメッセージを作成
        messages = [
            {"role": "system", "content": "あなたは経験豊富なビジネスコーチです。"},
            *chat_history,  # 会話履歴
            {"role": "user", "content": advice_prompt}
        ]

        print("Sending messages to OpenAI:", messages)  # OpenAIに送信するメッセージの確認

        # OpenAIのAPIを呼び出し
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        advice = response.choices[0].message.content
        print("Generated advice:", advice)  # 生成されたアドバイスの確認

        return {"advice": advice}

    except Exception as e:
        logger.error(f"Error generating advice: {str(e)}")
        print(f"Full error details: {str(e)}")  # 詳細なエラー情報の出力
        raise HTTPException(status_code=500, detail=str(e))

