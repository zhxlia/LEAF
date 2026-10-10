"""The fixed RF/XGBoost/TabPFN teacher average used during training."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .utils import normalize_probabilities

# The preprocessing classes below are retained from the existing implementation.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class OneHotFramePreprocessor:

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str], dense: bool=False):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.dense = dense
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean', keep_empty_features=True))])
        cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__', keep_empty_features=True)), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        self.ct = ColumnTransformer([('num', num_pipe, self.numeric_cols), ('cat', cat_pipe, self.categorical_cols)], remainder='drop')

    def fit(self, X: pd.DataFrame) -> 'OneHotFramePreprocessor':
        self.ct.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        out = self.ct.transform(X)
        if self.dense and hasattr(out, 'toarray'):
            out = out.toarray()
        return out

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

class DenseOrdinalPreprocessor:

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str]):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.numeric_fill_: dict[str, float] = {}
        self.category_maps_: dict[str, dict[str, int]] = {}
        self.cat_cardinalities_: list[int] = []

    def fit(self, X: pd.DataFrame) -> 'DenseOrdinalPreprocessor':
        self.numeric_fill_.clear()
        self.category_maps_.clear()
        self.cat_cardinalities_.clear()
        for col in self.numeric_cols:
            s = pd.to_numeric(X[col], errors='coerce')
            mean = float(s.mean()) if not np.isnan(s.mean()) else 0.0
            self.numeric_fill_[col] = mean
        for col in self.categorical_cols:
            values = X[col].astype('string').fillna('__MISSING__')
            uniq = pd.Index(values.unique().tolist())
            mapping = {str(v): i for (i, v) in enumerate(uniq.tolist())}
            if '__MISSING__' not in mapping:
                mapping['__MISSING__'] = len(mapping)
            self.category_maps_[col] = mapping
            self.cat_cardinalities_.append(len(mapping))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric_cols:
            num_df = pd.DataFrame(index=X.index)
            for col in self.numeric_cols:
                num_df[col] = pd.to_numeric(X[col], errors='coerce').fillna(self.numeric_fill_.get(col, 0.0))
            parts.append(num_df[self.numeric_cols].to_numpy(dtype=float))
        if self.categorical_cols:
            cat_df = pd.DataFrame(index=X.index)
            for col in self.categorical_cols:
                mapping = self.category_maps_[col]
                series = X[col].astype('string').fillna('__MISSING__').map(lambda z: mapping.get(str(z), mapping['__MISSING__']))
                cat_df[col] = series.astype(int)
            parts.append(cat_df[self.categorical_cols].to_numpy(dtype=float))
        if not parts:
            return np.zeros((len(X), 0), dtype=float)
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def align_probabilities(probabilities, encoded_classes, n_classes):
    """Align integer-encoded teacher classes with the forest class order."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(encoded_classes):
        raise ValueError("Teacher probability shape does not match class metadata")
    aligned = np.zeros((len(probabilities), n_classes), dtype=float)
    for column, cls in enumerate(encoded_classes):
        index = int(cls)
        if index < 0 or index >= n_classes or index != cls:
            raise ValueError("Unexpected encoded class from teacher")
        aligned[:, index] = probabilities[:, column]
    return normalize_probabilities(aligned)

class TeacherEnsemble:
    """All three teachers are required and are averaged with equal weights."""
    def __init__(self, task, seed, labels=None, tabpfn_checkpoint=None, device="auto"):
        self.task = task
        self.seed = seed
        self.labels = labels
        self.tabpfn_checkpoint = tabpfn_checkpoint
        self.device = device
        self.models = []
        self.preprocessors = []

    def fit(self, X, y):
        from xgboost import XGBClassifier, XGBRegressor
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        numeric = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
        categorical = [c for c in X if c not in numeric]
        if self.task == "classification":
            label_index = {label: i for i, label in enumerate(self.labels)}
            y_fit = np.asarray([label_index[label] for label in y], dtype=int)
            if len(np.unique(y_fit)) != len(self.labels):
                raise ValueError("Each class must be represented in the labeled set")
            rf_cls, xgb_cls, pfn_cls = RandomForestClassifier, XGBClassifier, TabPFNClassifier
        else:
            y_fit = np.asarray(y, dtype=float)
            rf_cls, xgb_cls, pfn_cls = RandomForestRegressor, XGBRegressor, TabPFNRegressor
        rf = rf_cls(n_estimators=300, max_depth=8, min_samples_leaf=1,
                    n_jobs=4, random_state=self.seed)
        xgb = xgb_cls(n_estimators=300, max_depth=3, learning_rate=0.05,
                      tree_method="hist", n_jobs=4, random_state=self.seed,
                      eval_metric="mlogloss" if self.task == "classification" else "rmse")
        pfn_kwargs = {"device": self.device, "ignore_pretraining_limits": True,
                      "random_state": self.seed}
        if not self.tabpfn_checkpoint:
            raise ValueError("Set a v2.5 default TabPFN checkpoint for the selected task")
        if self.tabpfn_checkpoint:
            path = Path(self.tabpfn_checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"TabPFN checkpoint not found: {path}")
            pfn_kwargs["model_path"] = str(path)
        pfn = pfn_cls(**pfn_kwargs)
        self.models = [rf, xgb, pfn]
        self.preprocessors = [
            OneHotFramePreprocessor(numeric, categorical),
            OneHotFramePreprocessor(numeric, categorical),
            DenseOrdinalPreprocessor(numeric, categorical),
        ]
        for model, preprocessor in zip(self.models, self.preprocessors):
            model.fit(preprocessor.fit_transform(X), y_fit)
        return self

    def predict(self, X):
        if len(self.models) != 3:
            raise RuntimeError("The three-teacher ensemble has not been fitted")
        predictions = []
        for model, preprocessor in zip(self.models, self.preprocessors):
            values = preprocessor.transform(X)
            if self.task == "classification":
                prediction = align_probabilities(model.predict_proba(values),
                                                  model.classes_, len(self.labels))
            else:
                prediction = np.asarray(model.predict(values), dtype=float).reshape(-1)
                if not np.isfinite(prediction).all():
                    raise ValueError("Non-finite regression teacher prediction")
            predictions.append(prediction)
        result = np.mean(np.stack(predictions), axis=0)
        return normalize_probabilities(result) if self.task == "classification" else result
