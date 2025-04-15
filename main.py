import torch
import psycopg2
from transformers import pipeline, logging

# --- Config ---
DB_CONFIG = {
    "dbname": "jeopardy",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432,
}

categories = [
    "SCIENCE",
    "HISTORY",
    "GEOGRAPHY",
    "LITERATURE",
    "POP CULTURE",
    "SPORTS",
    "ART",
    "LANGUAGE",
    "RELIGION & MYTHOLOGY",
    "POLITICS",
    "BUSINESS & ECONOMICS",
    "TECHNOLOGY",
    "FOOD & DRINK",
    "MISCELLANEOUS",
]

# --- Torch Info ---
print(f"Running torch {torch.__version__}")
print(f"- Cuda available: {torch.cuda.is_available()}")
print(f"- Devices available: {torch.cuda.device_count()}")
print(f"- Running on device: {torch.cuda.get_device_name(0)}")

logging.set_verbosity_error()

# --- Classifier Setup ---
classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=0
)


# --- DB Connection ---
def fetch_questions(limit=100):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT category, prompt, answer 
        FROM questions 
        WHERE prompt IS NOT NULL AND answer IS NOT NULL 
        LIMIT %s
    """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"category": r[0], "prompt": r[1], "answer": r[2]} for r in rows]


# --- Main Execution ---
questions = fetch_questions(limit=10)

for q in questions:
    text = f"Prompt: {q['prompt']}, Correct answer: {q['answer']}"
    result = classifier(q["prompt"], candidate_labels=categories)
    predicted_category = result["labels"][0]
    confidence = result["scores"][0]

    print("—" * 60)
    print(f"Prompt   : {q['prompt']}")
    print(f"Answer   : {q['answer']}")
    print(f"Actual   : {q['category']}")
    print(f"Predicted: {predicted_category} (Confidence: {confidence:.4f})")
