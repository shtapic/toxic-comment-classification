import shap


def shap_model(model, vectorizer, train_df):
    explanier = shap.LinearExplainer(model.estimators_[0], 
                vectorizer.transform(train_df['comment_text'].iloc[:1000]))

    shap_val = explanier.shap_values(
        vectorizer.transform(train_df['comment_text'].iloc[:1000])
    )

    shap.summary_plot(shap_val, 
        vectorizer.transform(train_df['comment_text'].iloc[:1000]).toarray(),
        feature_names=vectorizer.get_feature_names_out()
    )