"""Model training functions for churn prediction."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_baseline_models():
    """
    Returns a dictionary of unfitted baseline models for churn prediction.

    These models are NOT yet trained. They must undergo preprocessing and
    training (fit) before use for predictions or evaluation.

    Returns:
        dict: Dictionary mapping model names to unfitted sklearn model objects:
            - "Majority Class": DummyClassifier(strategy='most_frequent')
            - "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
            - "Decision Tree": DecisionTreeClassifier(random_state=42)
            - "Random Forest": RandomForestClassifier(random_state=42)
    """
    return {
        "Majority Class": DummyClassifier(strategy='most_frequent'),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }


def get_advanced_models():
    """
    Returns a dictionary of unfitted advanced models for churn prediction.

    These models are NOT yet trained. They must undergo preprocessing and
    training (fit) before use for predictions or evaluation.

    Returns:
        dict: Dictionary mapping model names to unfitted sklearn model objects:
            - "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            - "Random Forest (tuned depth)": RandomForestClassifier(random_state=42, max_depth=10, min_samples_leaf=5)
    """
    return {
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Random Forest (tuned depth)": RandomForestClassifier(random_state=42, max_depth=10, min_samples_leaf=5),
    }
