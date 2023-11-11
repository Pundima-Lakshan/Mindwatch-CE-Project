from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
IMAGE = Image.open(requests.get(URL, stream=True).raw)
TEXT = "What are cats doing?"

# IMAGE = Image.fromarray(frame)
# IMAGE = image.convert("RGB")

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(IMAGE, TEXT, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
