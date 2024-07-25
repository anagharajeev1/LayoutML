from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
import torch
from PIL import Image
import pytesseract

# Load the tokenizer and model
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

# Load the document image
image_path = 'HDFC.png'  # Update this path to the correct location of your image
image = Image.open(image_path)

# Perform OCR
ocr_results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

# Extract texts and bounding boxes
texts = ocr_results['text']
boxes = []
for i in range(len(texts)):
    (x, y, w, h) = (ocr_results['left'][i], ocr_results['top'][i], ocr_results['width'][i], ocr_results['height'][i])
    boxes.append([x, y, x+w, y+h])

# Print extracted texts and bounding boxes
print("Extracted texts and bounding boxes:")
for text, box in zip(texts, boxes):
    print(f"Text: {text}, Box: {box}")

# Tokenize the text
encoding = tokenizer(texts, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
token_type_ids = encoding["token_type_ids"]
bbox = encoding["bbox"]

# Make predictions
outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, bbox=bbox)
logits = outputs.logits

# Process and print the results
predictions = torch.argmax(logits, dim=-1)
print("Predictions:")
print(predictions)
