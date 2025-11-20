from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=10
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_slice_metrics(df, feature, y, preds):
    """
    Compute performance metrics on slices of data based on categorical
    features.

    Inputs
    ------
    df : pd.DataFrame
        Original dataframe with categorical features.
    feature : str
        Name of the categorical feature to slice on.
    y : np.ndarray
        True labels.
    preds : np.ndarray
        Predicted labels.

    Returns
    -------
    slice_metrics : list of dict
        List of dictionaries containing slice name and metrics.
    """
    slice_metrics = []

    for category in df[feature].unique():
        mask = df[feature] == category
        y_slice = y[mask]
        preds_slice = preds[mask]

        if len(y_slice) > 0:
            precision, recall, fbeta = compute_model_metrics(
                y_slice, preds_slice
            )
            slice_metrics.append({
                'feature': feature,
                'category': category,
                'n_samples': len(y_slice),
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta
            })

    return slice_metrics
