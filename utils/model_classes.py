import random

class DummyModel:
    def predict(self, features):
        """
        Dummy prediction logic.
        features: Expected to be an array-like object.
        Returns: (label, confidence)
        """
        # Simulate prediction logic
        # We don't use features, so we don't care about its type (numpy vs list)
        label = random.choice(["AI_GENERATED", "HUMAN"])
        confidence = random.uniform(0.5, 0.99)
        return label, confidence
