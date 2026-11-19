
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class InvalidZeroToNaN(BaseEstimator, TransformerMixin):

    invalid_columns = [
        "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI"
    ]

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame")

        missing = set(self.invalid_columns) - set(X.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(
            X,
            columns=self.feature_names_in_
        ).copy()

        for col in self.invalid_columns:
            X.loc[X[col] == 0, col] = np.nan

        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class GlucoseBMIFeature(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        if np.asarray(X).ndim != 2 or np.asarray(X).shape[1] != 8:
            raise ValueError("Expected exactly 8 input features")

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        glucose = X[:, 1]
        bmi = X[:, 5]

        interaction = glucose * bmi

        return np.column_stack([X, interaction])

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            input_features = [
                "Pregnancies", "Glucose", "BloodPressure",
                "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigreeFunction", "Age"
            ]

        return np.asarray(
            list(input_features) + ["Glucose_BMI_Index"],
            dtype=object
        )
