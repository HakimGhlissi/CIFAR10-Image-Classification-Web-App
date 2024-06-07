from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
from keras.models import load_model

app = Flask(__name__)

# Load the models
models = {
    'cnn_model': load_model('cnn_model.h5'),
    'cnn_model_best': load_model('cnn_model_best.h5'),
    # 'vgg_cifar10_optimized': load_model('vgg_cifar10_optimized.h5')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the file from the POST request
    file = request.files['file']
    model_name = request.form['model']
    
    # Extract the filename
    filename = file.filename
    
    # Save the file temporarily
    # file_path = f'./{filename}'
    # file.save(file_path)
    
    # Preprocess the image
    image = load_img(filename, target_size=(32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    
    # Select the model based on the user's choice
    model = models.get(model_name)
    if model is None:
        return jsonify({'error': 'Model not found'}), 400
    
    # Make prediction
    predictions = model.predict(image)
    class_idx = np.argmax(predictions, axis=1)[0]
    
    # CIFAR-10 class names
    class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]
    
    result = class_names[class_idx]
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
