# DataSciencce_task3
# Iris Flower Classification 

This is a simple Flask-based web application that predicts the species of an Iris flower based on four input features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The application uses a trained machine learning model (Random Forest Classifier) on the classic Iris dataset.

## 🔍 About the Dataset
The Iris dataset is one of the most famous datasets in machine learning. It contains 150 records under the following classes:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

Each record consists of 4 features (sepal length, sepal width, petal length, petal width) and a target label (species).

## 🛠 Technologies Used
- Python
- Flask
- scikit-learn
- HTML/CSS (Bootstrap 5)
- pandas

## 🚀 Getting Started



### 1. Install Dependencies
 manually:
```bash
pip install flask pandas scikit-learn joblib
```

### 2. Train and Save the Model
Use your existing training script or run:
```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("IRIS.csv")
le = LabelEncoder()
df["species_encoded"] = le.fit_transform(df["species"])
X = df.drop(["species", "species_encoded"], axis=1)
y = df["species_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "iris_model.pkl")
joblib.dump(le, "label_encoder.pkl")
```

### 3. Run the App
```bash
python app.py
```

Open your browser and go to: `http://127.0.0.1:5000`

## 📂 Project Structure
```
iris-flower-classifier/
├── app.py
├── iris_model.pkl
├── label_encoder.pkl
├── IRIS.csv
├── templates/
│   └── index.html
└── README.md
```

## 📸 UI Preview
- A clean Bootstrap form to enter sepal/petal dimensions
- Real-time species prediction result displayed below the form
  
![image](https://github.com/user-attachments/assets/7baf50a2-699d-4adc-94e1-265ed0dbc076)

![image-1](https://github.com/user-attachments/assets/0e718159-490b-42d5-be6b-ca57a81c0eec)

  

## 🧠 Model Used
- Random Forest Classifier
- Achieves >95% accuracy on test set

- ![WhatsApp Image 2025-04-20 at 15 20 09_feb0f104](https://github.com/user-attachments/assets/d451d61a-5af0-4943-899e-a5cc3adfb922)


## 📃 License
This project is open-source and free to use for learning and development.

