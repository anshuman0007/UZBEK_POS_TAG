import os
from src.data_loader import load_data
from src.train import train_model
import joblib

data_path = os.path.join('data', 'uzb.txt')
sentences, tags = load_data(data_path)

crf_model, X_test, y_test = train_model(sentences, tags)

# Save the model
joblib.dump(crf_model, 'models/pos_tagger_model.pkl')
joblib.dump(X_test, 'models/X_test.pkl')
joblib.dump(y_test, 'models/y_test.pkl')