import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DefectCSV:
    def __init__(self, file):
        self.file = file
        self.df = self.load_data()

    def load_data(self):
        # Load the CSV file from the uploaded file
        return pd.read_csv(self.file)

    def train_model(self):
        # Assume 'DefectStatus' is the target column
        X = self.df.drop(columns=['DefectStatus'])
        y = self.df['DefectStatus']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = XGBClassifier()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return model, accuracy

def train_model(uploaded_files):
    if uploaded_files:
        # Train the model using the first uploaded file (adjust if needed)
        defect_csv = DefectCSV(uploaded_files[0])
        model, accuracy = defect_csv.train_model()
        st.success(f"Model trained with accuracy: {accuracy}")
