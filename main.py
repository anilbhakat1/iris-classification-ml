"""
Iris Flower Classification using Scikit-learn
---------------------------------------------
Loads the public Iris dataset, splits it into training/testing sets,
trains a K-Nearest Neighbors (KNN) classifier, and reports evaluation
metrics including test accuracy and the confusion matrix.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    # 1. Load the public Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("IRIS FLOWER CLASSIFICATION")
    print("=" * 40)
    print(f"Number of samples : {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Classes           : {', '.join(iris.target_names)}")

    # 2. Split data: 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 3. Build and train a KNN classification pipeline.
    # Scaling is important for distance-based models such as KNN.
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    )
    model.fit(X_train, y_train)

    # 4. Make predictions on unseen test data
    y_pred = model.predict(X_test)

    # 5. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("\nMODEL: K-NEAREST NEIGHBORS (KNN)")
    print("-" * 40)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Test accuracy   : {accuracy:.2%}")

    print("\nConfusion Matrix")
    print("(Rows = actual class, columns = predicted class)")
    print(matrix)

    print("\nClassification Report")
    print(classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
    ))


if __name__ == "__main__":
    main()
