import joblib
import numpy as np

def predict_pos_tags(sentence):
    crf_model = joblib.load('models/pos_tagger_model.pkl')
    
    # Preprocess the input sentence into features
    features = [{'word': word} for word in sentence.split()]

    # Predict using the model
    prediction = crf_model.predict([features])
    
    return prediction[0]  # Return the predicted tags
