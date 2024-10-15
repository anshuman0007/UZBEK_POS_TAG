from src.predict import predict_pos_tags

if __name__ == "__main__":
    sentence = input("Enter a sentence in Uzbek: ")
    predicted_tags = predict_pos_tags(sentence)
    print("Predicted POS tags:", predicted_tags)
