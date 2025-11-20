# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained to predict whether an individual's income exceeds $50,000 per year based on census data. The model was developed as part of a machine learning deployment pipeline demonstration.

- **Model Type**: Random Forest Classifier
- **Model Version**: 1.0.0
- **Training Algorithm**: scikit-learn RandomForestClassifier
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42
- **Features**: 14 input features including age, education, occupation, work class, and other demographic attributes

## Intended Use

This model is intended for educational and demonstration purposes to showcase:
- Machine learning model development and deployment
- MLOps best practices including DVC, FastAPI, and CI/CD
- Model performance evaluation and monitoring

**Primary Intended Uses:**
- Educational demonstration of ML deployment pipelines
- Income prediction for census analysis

**Out-of-Scope Uses:**
- Making actual hiring, lending, or other high-stakes decisions
- Production use without thorough validation and bias testing

## Training Data

The model was trained on the UCI Adult Census Income dataset (also known as "Census Income" dataset).

- **Source**: UCI Machine Learning Repository
- **Dataset**: Adult Census Income dataset
- **Size**: Approximately 32,000 training samples after 80/20 train-test split
- **Features**: 14 demographic and employment-related features
- **Target**: Binary classification (income <=50K or >50K)

**Categorical Features:**
- workclass, education, marital-status, occupation, relationship, race, sex, native-country

**Preprocessing:**
- One-hot encoding for categorical features
- Label binarization for target variable
- Train-test split with 80/20 ratio

## Evaluation Data

The model was evaluated on a held-out test set representing 20% of the original dataset.

- **Test Size**: Approximately 6,500 samples
- **Same preprocessing applied**: One-hot encoding and label binarization
- **Evaluation includes**: Overall metrics and slice-based performance analysis

## Metrics

The model is evaluated using three primary metrics:

- **Precision**: Measures the accuracy of positive predictions
- **Recall**: Measures the model's ability to find all positive cases
- **F1 Score**: Harmonic mean of precision and recall

**Overall Performance** (approximate, based on typical results):
- Precision: ~0.75-0.80
- Recall: ~0.55-0.65
- F1 Score: ~0.63-0.70

**Slice-Based Performance:**
Performance metrics are computed for slices of data based on categorical features (education, occupation, race, sex, etc.) to identify potential biases and performance disparities across demographic groups. See `slice_output.txt` for detailed slice performance.

## Ethical Considerations

**Potential Biases:**
- The model may reflect historical biases present in census data
- Certain demographic groups may have disparate performance metrics
- Features like race and sex could lead to discriminatory outcomes if used in high-stakes decisions

**Fairness Concerns:**
- Model performance should be evaluated across different demographic slices
- Special attention should be paid to protected attributes
- The model should not be used for automated decision-making without human oversight

**Data Privacy:**
- Census data is aggregated and anonymized
- No personally identifiable information (PII) is included
- Model predictions should not be used to identify individuals

## Caveats and Recommendations

**Limitations:**
- Model is trained on historical census data and may not reflect current economic conditions
- Performance varies across different demographic groups
- Binary classification may oversimplify income distribution

**Recommendations:**
1. **Do not use in production** for high-stakes decisions without thorough validation
2. **Monitor performance** across demographic slices regularly
3. **Consider fairness metrics** beyond accuracy (e.g., equal opportunity, demographic parity)
4. **Retrain regularly** with updated data to maintain relevance
5. **Implement human oversight** for any decision-making processes
6. **Conduct bias audits** before deploying in any real-world scenario
7. **Document limitations** when sharing or deploying the model

**Future Improvements:**
- Implement fairness-aware learning techniques
- Explore more sophisticated model architectures
- Add explainability features (SHAP, LIME)
- Conduct comprehensive bias and fairness testing
