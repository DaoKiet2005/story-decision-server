from fastapi import FastAPI
import random
from unidecode import unidecode
from rapidfuzz import fuzz

# ===== ML =====
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()

# ===== DATA (Dữ liệu không thay đổi) =====
funny_stories = [
    "Ba chàng ngốc đi mua trâu",
    "Thầy bói xem voi",
    "Quan xử kiện",
    "Cười ra nước mắt",
    "Anh Ba keo kiệt"
]

sad_stories = [
    "Chiếc lá cuối cùng",
    "Lão Hạc",
    "Chí Phèo",
    "Vợ nhặt",
    "Những ngày thơ ấu"
]

horror_stories = [
    "Ngôi nhà hoang",
    "Con ma dưới gầm giường",
    "Chuyến xe lúc nửa đêm",
    "Tiếng gõ cửa trong đêm",
    "Căn phòng số 13"
]

# ===== ML TRAIN (Không thay đổi) =====
train_sentences = [
    "ke truyen vui", "truyen hai huoc", "ke chuyen cuoi",
    "truyen buon", "truyen cam dong", "cau chuyen bi kich",
    "truyen ma", "kinh di", "chuyen rung ron"
]

train_labels = [
    "FUNNY_STORY", "FUNNY_STORY", "FUNNY_STORY",
    "SAD_STORY", "SAD_STORY", "SAD_STORY",
    "HORROR_STORY", "HORROR_STORY", "HORROR_STORY"
]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_sentences)

model = MultinomialNB()
model.fit(X_train, train_labels)

# ===== KEYWORDS (Không thay đổi) =====
KEYWORDS = {
    "FUNNY_STORY": ["hai", "vui", "cuoi"],
    "SAD_STORY": ["buon", "cam dong", "bi kich"],
    "HORROR_STORY": ["ma", "kinh di", "rung ron"]
}

# ===== API ĐIỀU CHỈNH =====
@app.post("/decision")
def decide_story(data: dict):
    raw_question = data.get("question", "")
    question = unidecode(raw_question.lower())

    decisions = []
    stories = {}
    
    # Tạo một mapping dễ tra cứu hơn
    ALL_STORIES = {
        "FUNNY_STORY": funny_stories,
        "SAD_STORY": sad_stories,
        "HORROR_STORY": horror_stories
    }

    # 🔹 Fuzzy keyword matching
    for intent, keys in KEYWORDS.items():
        for k in keys:
            if fuzz.partial_ratio(k, question) > 70:
                decisions.append(intent)
                break

    # 🔹 ML intent (fallback)
    X_test = vectorizer.transform([question])
    ml_intent = model.predict(X_test)[0]

    if ml_intent not in decisions:
        decisions.append(ml_intent)

    # 🔹 Pick stories: Trả về TOÀN BỘ danh sách truyện (Không dùng random.choice)
    for d in decisions:
        # Thay vì random.choice(list), ta gán cả list vào stories[d]
        stories[d] = ALL_STORIES.get(d, []) # Dùng .get(d, []) để tránh lỗi nếu intent không hợp lệ

    return {
        "question": raw_question,
        "decisions": decisions,
        "stories": stories
    }