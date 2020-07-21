from typing import NamedTuple

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

from classifiers import DecisionTreeClassifier, LogisticClassifier, RandomForestClassifier, SVMClassifier


class SplitData(NamedTuple):
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


@st.cache(persist=True)
def load_data() -> pd.DataFrame:
    data = pd.read_csv("/home/nastiositi/disk_e/PycharmProjects/Machine_Learning_App/mushrooms.csv")
    label_encoder = LabelEncoder()

    for col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])
    return data


@st.cache(persist=True)
def split(df: pd.DataFrame) -> SplitData:
    y = df.type
    x = df.drop(columns=['type'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return SplitData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test
    )


class_names = ['edible', 'poisonous']
split_data = split(load_data())


def plot_metrics(metrics_list, model):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, split_data.x_test, split_data.y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, split_data.x_test, split_data.y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        plot_precision_recall_curve(model, split_data.x_test, split_data.y_test)
        st.pyplot()


choose_classifier = {
        "Random Forest": RandomForestClassifier,
        "Logistic Regression": LogisticClassifier,
        "Support Vector Machine (SVM)": SVMClassifier,
        "Decision Tree Classifier": DecisionTreeClassifier,
}


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.subheader("Choose Classifier")

    classifiers = ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "Decision Tree Classifier")
    classifier = st.sidebar.selectbox("Classifier",
                                      classifiers)

    class_factory = choose_classifier[classifier]
    model, data = class_factory().model_class()

    metric_list = ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve')
    metrics = st.sidebar.multiselect("What metrics to plot?",
                                     metric_list)

    if st.sidebar.button("Classify", key='classify1'):
        st.subheader(f'{classifier} Results: ')
        model.fit(split_data.x_train, split_data.y_train)
        data.columns = ['Using these Parameters']
        st.table(data)
        accuracy = model.score(split_data.x_test, split_data.y_test)
        y_pred = model.predict(split_data.x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(split_data.y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(split_data.y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics, model)


if __name__ == '__main__':
    main()