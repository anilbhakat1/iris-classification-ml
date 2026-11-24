# Iris Flower Classification with Scikit-learn

A beginner-friendly machine learning project that classifies Iris flowers into
three species using the **K-Nearest Neighbors (KNN)** algorithm.

## Project overview

This project uses the public Iris dataset included in Scikit-learn. It:

1. Loads the dataset (150 samples and 4 numeric features).
2. Splits the data into 80% training data and 20% testing data.
3. Scales the features using `StandardScaler`.
4. Trains a KNN classifier with 5 neighbors.
5. Reports test accuracy, a confusion matrix, and a classification report.

## Dataset

The Iris dataset contains measurements of:
- Sepal length
- Sepal width
- Petal length
- Petal width

The three target classes are Setosa, Versicolor, and Virginica. The dataset is
bundled with Scikit-learn, so no separate download is required.

## Requirements

- Python 3.9 or newer
- pip

## Run locally

### 1. Extract or clone this repository

```bash
git clone https://github.com/YOUR-USERNAME/iris-classification-ml.git
cd iris-classification-ml
```

If you downloaded the ZIP, extract it and open a terminal in the project folder.

### 2. (Recommended) Create a virtual environment

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the program

```bash
python main.py
```

The program prints the test accuracy, confusion matrix, and classification report.
The exact accuracy is calculated when the program runs.

## Project structure

```text
iris-classification-ml/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Model evaluation

- **Accuracy:** proportion of test predictions that are correct.
- **Confusion matrix:** shows actual classes against predicted classes.
- **Precision, recall, and F1-score:** provide class-wise evaluation.

For the confusion matrix, rows represent actual classes and columns represent
predicted classes. Class order is Setosa, Versicolor, Virginica.

## Upload this project to GitHub

1. Sign in to GitHub and create a new repository named `iris-classification-ml`.
2. Keep the repository public if you want to share it as a portfolio project.
3. Do not initialize it with another README if you plan to push this folder as-is.
4. Open a terminal inside this project folder and run:

```bash
git init
git add .
git commit -m "Add Iris flower classification project"
git branch -M main
git remote add origin https://github.com/anilbhakat1/iris-classification-ml.git
git push -u origin main
```

anilbhakat1.

## Author

Anil Bhakat 
