import torch
from transformers import pipeline, logging


print(f"Running torch {torch.__version__}")
print(f"- Cuda available: {torch.cuda.is_available()}")
print(f"- Devices available: {torch.cuda.device_count()}")
print(f"- Running on device: {torch.cuda.get_device_name(0)}")

# set verbosity to error level
logging.set_verbosity_error()

# create classfier
classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=0
)

# TODO: replace this with question list from db
question = "This scientist developed the theory of relativity."

# possible categories
categories = ["SCIENCE", "HISTORY", "POP CULTURE", "SPORTS", "GEOGRAPHY", "LITERATURE"]

result = classifier(question, candidate_labels=categories)
scores = result["scores"]
max_score = max(scores)
max_index = scores.index(max_score)
print(categories[max_index])
