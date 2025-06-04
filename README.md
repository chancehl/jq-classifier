## Jeopardy Question Classifier

This project fetches Jeopardy questions from a PostgreSQL database and classifies them using a zero-shot text classifier from HuggingFace transformers. After classification the questions with predictions can be saved locally as JSON for later use.

### Running the script

```bash
python main.py
```

The script connects to the database configured in `main.py`, retrieves a batch of questions, classifies them, and then writes the results (including the predicted category) to `questions.json` while also printing each prediction to the console.

