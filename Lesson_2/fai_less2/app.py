import gradio as gr
import spaces
from fastai.vision.all import *

learn = load_learner('model.pkl')

categories = ('Digimon', 'Pokemon')

@spaces.GPU
def classify_image(img):
    img = PILImage.create(img)
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(type="pil")
label = gr.Label()

examples = [
    '120px-Agumon_black.jpg',
    'charmander.png'
]

intf = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    examples=examples
)

intf.launch()