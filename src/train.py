from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split

def sent_to_features(sent):
    return [{'word': word} for word in sent]

def sent_to_labels(tags):
    return tags

def train_model(sentences, tags):
    X = [sent_to_features(s) for s in sentences]
    y = [sent_to_labels(t) for t in tags]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    crf = CRF(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True)
    crf.fit(X_train, y_train)

    return crf, X_test, y_test
