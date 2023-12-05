from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image # to change img path to actual image
import numpy as np

class_names = {
    0:'airplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck',
}

model = models.load_model("baseline_mariya.keras")

def image_predict(model, path_to_img):
    img = Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((32,32))
    data = np.asarray(img)
    data = data/255
    probs = model.predict(np.array([data])[:1])
    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]

    return top_prob,top_pred


# index = "<h1>Kaise Ho!!</h1>"
# index = "# Kaise Ho!!"

img_path = "img.png"
content = ""
prob = 0
pred = ""

index = """ 
<|text-center|
<|{"logo.png"}|image|width=12vw|>

<|{content}|file_selector|extensions=.png,.jpeg|>
Select an image from your file system


<|{pred}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name,var_value):
    if var_name == "content":
        top_prob,top_pred = image_predict(model,var_value)
        state.prob = round(top_prob*100)
        state.pred = "This is a " + top_pred
        state.img_path = var_value
    # print(var_name, var_value)


app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)  # means we dont need to type python classifier.py to run it again just refresh the page
