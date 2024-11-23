########################################################
## This file contains the code for the Flask web app  ##
########################################################

import torch
from utils.model_utils import load_model
from utils.data_utils import return_kmer, is_dna_sequence
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__, template_folder="templates")

model_config = {
    "model_path": "results/classification/model", # Path to the trained model
    "num_classes": 6,
}

# model, tokenizer, device = load_model(model_config, return_model=True)

model, tokenizer = load_model(model_config, return_model=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dictionary to convert the predicted class by the model to the class name
class_names_dic = {
    1: "SARS-COV-1",
    2: "MERS",
    3: "SARS-COV-2",
    4: "Ebola ",
    5: "Dengue",
    6: "Influenza",
}

KMER = 3
SEQ_MAX_LEN = 512

def huggingface_predict(sequence):
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to the appropriate device
    model.to(device)
    
    # Tokenize the input
    inputs = tokenizer(
        sequence,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Move predictions back to CPU for numpy operations
    predictions = outputs.logits.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # Get the predicted class and its probability
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_probability = probabilities[0][predicted_class]
    
    return predicted_class, class_probability

@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST']) # handle the post request from the form in index.html
def predict():
    input = request.form['input_sequence']
    prediction, probability = huggingface_predict(input)
    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)