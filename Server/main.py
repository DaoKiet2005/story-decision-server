from fastapi import FastAPI
from unidecode import unidecode
from rapidfuzz import fuzz

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI(title="Anime Decision API")

STORIES = {
    "Attack on Titan": [
        "attack on titan", "aot", "shingeki no kyojin", "snk",
        "dai chien nguoi khong lo", "eren", "levi", "titan", "paradis"
    ],
    "Black Butler": [
        "black butler", "kuroshitsuji", "sebastian", "ciel", "hac quan gia"
    ],
    "Black Clover": [
        "black clover", "asta", "yuno", "ma phap", "black bull"
    ],
    "Chainsaw Man": [
        "chainsaw man", "nguoi cua", "denji", "makima", "quy"
    ],
    "Doraemon": [
        "doraemon", "chu meo may", "nobita", "bao boi"
    ],
    "Dragon Ball": [
        "dragon ball", "7 vien ngoc rong", "goku", "vegeta", "saiyan"
    ],
    "Hunter x Hunter": [
        "hunter x hunter", "hxh", "gon", "killua", "bang nhen"
    ],
    "Jujutsu Kaisen": [
        "jujutsu kaisen", "jjk", "chu thuat", "gojo", "sukuna"
    ],
    "Mob Psycho 100": [
        "mob psycho", "mob", "sieu nang luc", "reigen"
    ],
    "My Hero Academia": [
        "my hero academia", "mha", "hoc vien anh hung", "deku", "all might"
    ],
    "Naruto": [
        "naruto", "hokage", "lang la", "sasuke", "akatsuki"
    ],
    "One Piece": [
        "one piece", "dao tac", "luffy", "zoro", "trai ac quy"
    ],
    "Detective Conan": [
        "conan", "tham tu", "kaito kid", "to chuc ao den"
    ],
    "Demon Slayer": [
        "kimetsu no yaiba", "demon slayer", "diet quy", "tanjiro", "muzan"
    ],
    "Sailor Moon": [
        "sailor moon", "thuy thu mat trang", "usagi"
    ],
    "Tokyo Ghoul:re": [
        "tokyo ghoul", "nga quy", "kaneki", "ccg"
    ],
    "Vinland Saga": [
        "vinland saga", "viking", "thorfinn", "chien binh"
    ]
}

train_sentences = [
    "truyen hai huoc", "truyen vui",
    "truyen buon", "truyen cam dong",
    "truyen ma", "kinh di", "rung ron"
]

train_labels = [
    "FUNNY", "FUNNY",
    "SAD", "SAD",
    "HORROR", "HORROR", "HORROR"
]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_sentences)

model = MultinomialNB()
model.fit(X_train, train_labels)

@app.post("/decision")
def decide_story(data: dict):
    raw_question = data.get("question", "")
    question = unidecode(raw_question.lower())

    scores = {}

    for story, keywords in STORIES.items():
        best_score = 0
        for k in keywords:
            if len(k) < 4: 
                continue
            score = fuzz.partial_ratio(k, question)
            best_score = max(best_score, score)

        if best_score >= 75:
            scores[story] = best_score

    if scores:
        best_story = max(scores, key=scores.get)
        return {
            "question": raw_question,
            "decision_type": "EXACT_STORY_MATCH",
            "story": best_story,
            "confidence": scores[best_story]
        }

    GENRE_RULES = {
        "hai": ["Doraemon", "Spy x Family"],
        "vui": ["Doraemon"],
        "kinh di": ["Tokyo Ghoul:re", "Chainsaw Man"],
        "hanh dong": ["Attack on Titan", "Naruto", "One Piece"]
    }

    for key, stories in GENRE_RULES.items():
        if key in question:
            return {
                "question": raw_question,
                "decision_type": "GENRE_RULE",
                "stories": stories
            }

    X_test = vectorizer.transform([question])
    intent = model.predict(X_test)[0]

    ML_MAP = {
        "FUNNY": ["Doraemon"],
        "SAD": ["Attack on Titan", "Vinland Saga"],
        "HORROR": ["Tokyo Ghoul:re", "Chainsaw Man"]
    }

    return {
        "question": raw_question,
        "decision_type": "ML_FALLBACK",
        "intent": intent,
        "stories": ML_MAP.get(intent, [])
    }
