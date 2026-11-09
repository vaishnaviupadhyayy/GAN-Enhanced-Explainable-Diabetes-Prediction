import pandas as pd
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
if __name__ == "__main__":
    try:
        df = pd.read_csv("diabetes.csv")
        print("diabetes.csv already exists:", df.shape)
    except FileNotFoundError:
        print("Downloading Pima Indians Diabetes dataset...")
        df = pd.read_csv(URL, header=None, names=COLUMNS)
        df.to_csv("diabetes.csv", index=False)
        print("Saved diabetes.csv:", df.shape)
