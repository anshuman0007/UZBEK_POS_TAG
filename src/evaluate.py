import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report



X_test = joblib.load('models/X_test.pkl')
y_test = joblib.load('models/y_test.pkl')

# Load the trained CRF model
crf_model = joblib.load('models/pos_tagger_model.pkl')

# Predict using the model
y_pred = crf_model.predict(X_test)

# Step 2: Use MultiLabelBinarizer to convert sequences
mlb = MultiLabelBinarizer()

# Transform y_test and y_pred into binary arrays
y_test_bin = mlb.fit_transform(y_test)
y_pred_bin = mlb.transform(y_pred)

# Evaluate the model
print(classification_report(y_test_bin, y_pred_bin))
