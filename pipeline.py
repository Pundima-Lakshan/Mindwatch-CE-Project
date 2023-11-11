"""
docs
"""

from transformers import pipeline
from PIL import Image
from transformers import ViltForQuestionAnswering

IMG_PATH = "D:/Projects/1 CEProject/resources/images/WhatsApp Image 2023-10-04 at 16.27.58_41bd65db.jpg"
MODEL_PATH = "D:/Projects/1 CEProject/git/Mindwatch-CE-Project/vilt-b32-finetuned-vqa/"

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
oracle = pipeline("visual-question-answering", model=MODEL_PATH)
image = Image.open(IMG_PATH)
out = oracle(question="Is there a person?", image=image)

print(out)
