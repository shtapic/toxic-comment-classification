from turtle import pd
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, precision_score, recall_score, classification_report
import numpy as np
from matplotlib import pyplot as plt


def metrics_model(y_true, y_prob, thresholds=0.5):
    y_pred = applay_thresholds(y_prob, thresholds)

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    precision = average_precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print("--------------------------------")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("--------------------------------")
    print(classification_report(y_true, y_pred))
    
    # return f1_macro, accuracy, precision, recall


def applay_thresholds(y_prob, thresholds):
    if not isinstance(thresholds, list):
        y_pred = y_prob > thresholds
        return y_pred.astype(int)

    y_pred = np.zeros(y_prob.shape)
    for i, threshold in enumerate(thresholds):
        y_pred[:, i] = y_prob[:, i] > threshold
    return y_pred.astype(int)


def get_top_tox(vectorizer, model, top_k=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.estimators_[0].coef_.flatten()

    top_toxic = feature_names[np.argsort(coefs)][-top_k:][::-1]
    top_non_toxic = feature_names[np.argsort(coefs)[:top_k]]

    print("TOXIC WORDS:")
    print(top_toxic)

    print("\nNON-TOXIC WORDS:")
    print(top_non_toxic)

    plt.figure(figsize=(8, 4))
    plt.barh(top_toxic, coefs[np.argsort(coefs)][-top_k:][::-1])
    plt.xlabel("Coefficient Toxicity Value")
    plt.title("Top Toxic Words")
    plt.show()


    # return top_toxic, top_non_toxic, coefs[np.argsort(coefs)][-top_k:][::-1], coefs[np.argsort(coefs)[:top_k]]



def explain_text(text, vectorizer, model, top_k=5):
    vec = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.estimators_[0].coef_.flatten()

    contributions = vec.toarray()[0] * coefs
    idx = np.argsort(contributions)[-top_k:]

    print( pd.DataFrame({
        "word": feature_names[idx],
        "contribution": contributions[idx]
    }).sort_values("contribution", ascending=False)
    )
        

def word_contributions(text, vectorizer, model):
    clf = model.estimators_[0]

    words = text.lower().split()
    w = clf.coef_[0]

    contributions = {}

    for word in words:
        if word in vectorizer.model.wv:
            vec = vectorizer.model.wv[word]
            contributions[word] = float(np.dot(w, vec))

    print (dict(
        sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    ))
