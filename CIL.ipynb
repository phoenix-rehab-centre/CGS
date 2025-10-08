from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def predict_image(image):
    model = load_model('best_model.h5')
    img = Image.open(image).convert('L').resize((28,28))
    x = np.array(img).astype('float32') / 255.0
    x = x.reshape(1,28,28,1)
    probs = model.predict(x)[0]
    pred = np.argmax(probs)
    confidence = np.max(probs)
    print(pred)
    print(confidence)

predict_image("2.png")
