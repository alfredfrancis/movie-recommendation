import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("./bert-imdb")
tokenizer = BertTokenizer.from_pretrained("./bert-imdb")

# Inference function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "positive" if prediction == 1 else "negative"

review = """
"Empuraan" is not just a sequel to "Lucifer"—it's a statement. Prithviraj Sukumaran takes everything that worked in the first film and amplifies it, delivering a cinematic masterpiece that redefines Malayalam cinema. This isn't just a mass entertainer; it's a high-stakes political thriller woven with action, intrigue, and breathtaking storytelling.
"""
# Example
print(predict_sentiment(review))
