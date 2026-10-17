from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Titanic-Dataset.csv"
MODEL_PATHS = {
    "Logistic Regression": BASE_DIR / "lregration.pkl",
    "K-Nearest Neighbors": BASE_DIR / "knn.pkl",
    "Support Vector Machine": BASE_DIR / "svm.pkl",
}

st.set_page_config(page_title="Titanic Survival Explorer", page_icon="🚢", layout="wide")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models() -> dict[str, object]:
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    prepared = data.copy()
    prepared["Age"] = prepared["Age"].fillna(prepared["Age"].mean())
    prepared["Embarked"] = prepared["Embarked"].fillna(prepared["Embarked"].mode()[0])
    prepared["FamilySize"] = prepared["SibSp"] + prepared["Parch"] + 1
    prepared["IsAlone"] = (prepared["FamilySize"] == 1).astype(int)
    prepared = prepared.drop(
        columns=["Cabin", "Ticket", "PassengerId", "Name"]
    )
    prepared = pd.get_dummies(
        prepared,
        columns=["Sex", "Embarked"],
        drop_first=True,
        dtype=int,
    )
    features = prepared.drop(columns="Survived")
    target = prepared["Survived"]
    return features, target


def passenger_features(
    passenger_class: int,
    age: int,
    siblings_spouses: int,
    parents_children: int,
    fare: float,
    sex: str,
    embarked: str,
    feature_names: list[str],
) -> pd.DataFrame:
    family_size = siblings_spouses + parents_children + 1
    values = {
        "Pclass": passenger_class,
        "Age": age,
        "SibSp": siblings_spouses,
        "Parch": parents_children,
        "Fare": fare,
        "FamilySize": family_size,
        "IsAlone": int(family_size == 1),
        "Sex_male": int(sex == "Male"),
        "Embarked_Q": int(embarked == "Queenstown (Q)"),
        "Embarked_S": int(embarked == "Southampton (S)"),
    }
    return pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)


st.title("🚢 Titanic Survival Explorer")
st.caption("Explore the passenger data, compare the saved classifiers, and try a prediction.")

dataset = load_dataset()
models = load_models()
model_feature_names = list(next(iter(models.values())).feature_names_in_)
features, target = prepare_features(dataset)
features = features.reindex(columns=model_feature_names, fill_value=0)
_, x_test, _, y_test = train_test_split(
    features, target, test_size=0.20, random_state=42
)

survivor_count = int(dataset["Survived"].sum())
survival_rate = float(dataset["Survived"].mean() * 100)
overview = st.columns(3)
overview[0].metric("Passengers", f"{len(dataset):,}")
overview[1].metric("Survivors", f"{survivor_count:,}")
overview[2].metric("Overall survival rate", f"{survival_rate:.1f}%")

data_tab, models_tab, predictor_tab = st.tabs(
    ["Passenger insights", "Model comparison", "Try a prediction"]
)

with data_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Survival rate by passenger class")
        by_class = (
            dataset.groupby("Pclass", observed=True)["Survived"]
            .mean()
            .mul(100)
            .rename("Survival rate (%)")
            .to_frame()
        )
        st.bar_chart(by_class)
    with right:
        st.subheader("Survival rate by sex")
        by_sex = (
            dataset.groupby("Sex", observed=True)["Survived"]
            .mean()
            .mul(100)
            .rename("Survival rate (%)")
            .to_frame()
        )
        st.bar_chart(by_sex)

    st.subheader("Passenger age distribution")
    age_data = dataset["Age"].fillna(dataset["Age"].median())
    age_bins = pd.cut(age_data, bins=range(0, 91, 10), right=False)
    age_counts = age_bins.value_counts(sort=False).rename("Passengers").to_frame()
    age_counts.index = age_counts.index.astype(str)
    st.bar_chart(age_counts)
    st.caption("The dataset contains missing ages; the chart fills those with the median for display.")

with models_tab:
    metric_rows = []
    for name, model in models.items():
        predictions = model.predict(x_test)
        metric_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index("Model")
    st.subheader("Saved model performance")
    st.caption("Metrics are evaluated on the same 20% holdout split used when these models were trained.")
    st.bar_chart(metrics)
    st.dataframe(metrics.style.format("{:.1%}"), width="stretch")

with predictor_tab:
    st.subheader("Passenger details")
    with st.form("passenger_details"):
        first, second, third = st.columns(3)
        with first:
            passenger_class = st.selectbox("Passenger class", [1, 2, 3], index=2)
            age = st.slider("Age", min_value=0, max_value=100, value=30)
            sex = st.selectbox("Sex", ["Female", "Male"])
        with second:
            siblings_spouses = st.number_input(
                "Siblings / spouses aboard", min_value=0, max_value=10, value=0
            )
            parents_children = st.number_input(
                "Parents / children aboard", min_value=0, max_value=10, value=0
            )
            embarked = st.selectbox(
                "Port of embarkation",
                ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"],
                index=2,
            )
        with third:
            fare = st.number_input(
                "Fare (£)", min_value=0.0, max_value=600.0, value=32.0, step=1.0
            )
            selected_model = st.selectbox("Classifier", list(models))
            submitted = st.form_submit_button("Predict survival", type="primary")

    if submitted:
        model = models[selected_model]
        model_features = passenger_features(
            passenger_class,
            age,
            int(siblings_spouses),
            int(parents_children),
            fare,
            sex,
            embarked,
            model_feature_names,
        )
        prediction = int(model.predict(model_features)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            class_index = list(model.classes_).index(1)
            probability = float(model.predict_proba(model_features)[0][class_index])

        if prediction:
            st.success("This model predicts that the passenger would survive.")
        else:
            st.warning("This model predicts that the passenger would not survive.")
        if probability is not None:
            st.metric("Estimated survival probability", f"{probability:.1%}")
            st.progress(probability)

st.divider()
st.caption("Predictions are outputs from the trained models and are not guarantees of real-world outcomes.")
