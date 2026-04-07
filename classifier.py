"""
classifier.py — Machine Learning for Motor Imagery Classification.

This module handles:
  - Training a CSP+LDA classifier on your EEG data
  - Evaluating accuracy with cross-validation
  - Saving/loading trained models
  - Real-time prediction with confidence smoothing

=== WHY CSP + LDA? ===

This combination is the gold standard for motor imagery BCI because:

1. CSP (Common Spatial Pattern) creates optimal spatial filters that
   maximize the difference between "rest" and "imagine" conditions.
   This is the feature extraction step.

2. LDA (Linear Discriminant Analysis) finds the optimal linear boundary
   between classes. It's fast (microseconds to predict), robust with
   small training sets, and well-understood theoretically.

3. Together, CSP+LDA has been shown in dozens of studies to achieve
   70-85% accuracy for 2-class motor imagery — which is sufficient
   for reliable BCI control.

=== WHY NOT DEEP LEARNING? ===

For EOH, CSP+LDA is better because:
  - Works with ~30 trials per class (deep learning needs thousands)
  - You can explain exactly what it's doing to judges
  - Trains in < 1 second
  - No GPU needed
  - Published, validated, reproducible

You can mention deep learning as "future work" on your poster.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from collections import deque
import joblib
import os
import config
from bci.features import CSPFilter


class BCIClassifier:
    """
    Motor imagery classifier using CSP spatial filters + LDA.
    
    Training workflow:
        clf = BCIClassifier()
        clf.train(epochs, labels)    # epochs shape: (n, ch, samples)
        clf.save()                   # save to disk
    
    Prediction workflow:
        clf = BCIClassifier()
        clf.load()                   # load from disk
        label, confidence = clf.predict_single(epoch)  # one epoch
    """
    
    def __init__(self):
        self.csp = CSPFilter(n_components=config.CSP_N_COMPONENTS)
        self.lda = LinearDiscriminantAnalysis()
        self.is_trained = False
        
        # Smoothing buffer for real-time predictions
        self._prediction_buffer = deque(
            maxlen=config.PREDICTION_SMOOTHING
        )
    
    def train(self, epochs, labels):
        """
        Train the CSP+LDA classifier.
        
        Args:
            epochs: np.ndarray, shape (n_epochs, n_channels, n_samples)
                   Preprocessed EEG epochs.
            labels: np.ndarray, shape (n_epochs,)
                   Class labels (0=REST, 1=CLOSE, etc.)
        
        Returns:
            dict with training results (accuracy, confusion matrix, etc.)
        
        WHAT HAPPENS:
          1. CSP learns spatial filters from the training data
          2. CSP transforms epochs → feature vectors
          3. LDA learns the decision boundary
          4. Cross-validation estimates real-world accuracy
        """
        print(f"\n{'='*60}")
        print(f"  TRAINING CSP+LDA CLASSIFIER")
        print(f"{'='*60}")
        print(f"  Epochs: {len(epochs)}")
        print(f"  Channels: {epochs.shape[1]}")
        print(f"  Samples per epoch: {epochs.shape[2]}")
        print(f"  Classes: {np.unique(labels)} → {config.CLASS_NAMES}")
        
        # Check class balance
        for c in np.unique(labels):
            n = np.sum(labels == c)
            print(f"  Class {c} ({config.CLASS_NAMES[c]}): {n} epochs")
        
        # Step 1: Fit CSP on all data
        self.csp.fit(epochs, labels)
        
        # Step 2: Extract features
        X = self.csp.transform(epochs)
        y = labels
        
        print(f"\n  Feature matrix: {X.shape}")
        
        # Step 3: Cross-validation (tells you expected real-world accuracy)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        cv_scores = cross_val_score(self.lda, X, y, cv=cv, scoring='accuracy')
        
        print(f"\n  Cross-validation accuracy: {cv_scores.mean():.1%} "
              f"(± {cv_scores.std():.1%})")
        print(f"  Per-fold: {[f'{s:.1%}' for s in cv_scores]}")
        
        # Step 4: Train final model on ALL data
        self.lda.fit(X, y)
        self.is_trained = True
        
        # Step 5: Full training set evaluation (for confusion matrix)
        y_pred = self.lda.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        print(f"\n  Training set confusion matrix:")
        print(f"  {cm}")
        print(f"\n  Classification report:")
        print(classification_report(
            y, y_pred,
            target_names=config.CLASS_NAMES[:len(np.unique(y))]
        ))
        
        # Interpret for the user
        accuracy = cv_scores.mean()
        if accuracy >= 0.75:
            print("  ✅ EXCELLENT — this should work well for the demo!")
        elif accuracy >= 0.65:
            print("  ✅ GOOD — reliable enough for controlled demos.")
        elif accuracy >= 0.55:
            print("  ⚠️  MARGINAL — collect more training data or check electrodes.")
        else:
            print("  ❌ POOR — check electrode placement and signal quality.")
            print("     Try: more trials, better electrode contact, less movement.")
        
        print(f"{'='*60}\n")
        
        return {
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'confusion_matrix': cm,
            'n_features': X.shape[1],
        }
    
    def predict_single(self, epoch):
        """
        Classify a single epoch in real-time.
        
        Args:
            epoch: np.ndarray, shape (n_channels, n_samples)
                  A single preprocessed EEG window.
        
        Returns:
            (predicted_class, confidence)
            - predicted_class: int (0=REST, 1=CLOSE, etc.)
            - confidence: float in [0, 1]. Higher = more certain.
        
        CONFIDENCE EXPLAINED:
          LDA outputs a probability for each class (using Bayes' rule
          with Gaussian class-conditional densities). The confidence
          is the probability of the predicted class.
          
          0.50 = pure guessing (for 2-class)
          0.55 = slightly above chance → set as threshold in config
          0.80+ = very confident
        """
        if not self.is_trained:
            raise RuntimeError("Call train() or load() first!")
        
        # Reshape for CSP: needs (1, n_channels, n_samples)
        epoch_3d = epoch[np.newaxis, :, :]
        
        # Extract features
        features = self.csp.transform(epoch_3d)  # shape: (1, n_features)
        
        # Predict
        proba = self.lda.predict_proba(features)[0]  # shape: (n_classes,)
        predicted = np.argmax(proba)
        confidence = proba[predicted]
        
        return predicted, confidence
    
    def predict_smoothed(self, epoch):
        """
        Classify with temporal smoothing — reduces jitter.
        
        Uses a rolling buffer of the last N predictions.
        The final output is the MAJORITY VOTE with averaged confidence.
        
        This is what you want for controlling the hand — it prevents
        the hand from flipping open/closed every 250ms due to noise.
        
        Args:
            epoch: np.ndarray, shape (n_channels, n_samples)
        
        Returns:
            (predicted_class, confidence, raw_class, raw_confidence)
        """
        raw_pred, raw_conf = self.predict_single(epoch)
        
        self._prediction_buffer.append((raw_pred, raw_conf))
        
        if len(self._prediction_buffer) < 2:
            return raw_pred, raw_conf, raw_pred, raw_conf
        
        # Majority vote
        preds = [p[0] for p in self._prediction_buffer]
        confs = [p[1] for p in self._prediction_buffer]
        
        # Count votes for each class
        classes, counts = np.unique(preds, return_counts=True)
        winner_idx = np.argmax(counts)
        smoothed_pred = classes[winner_idx]
        
        # Average confidence of the winning class
        winning_confs = [c for p, c in self._prediction_buffer if p == smoothed_pred]
        smoothed_conf = np.mean(winning_confs)
        
        return smoothed_pred, smoothed_conf, raw_pred, raw_conf
    
    def reset_smoothing(self):
        """Clear the prediction buffer (e.g., between demo runs)."""
        self._prediction_buffer.clear()
    
    def save(self, model_path=None, csp_path=None):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path for LDA model. Default from config.
            csp_path: Path for CSP filters. Default from config.
        """
        model_path = model_path or config.MODEL_FILE
        csp_path = csp_path or config.CSP_FILE
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(self.lda, model_path)
        joblib.dump(self.csp, csp_path)
        
        print(f"[Classifier] Model saved to {model_path}")
        print(f"[Classifier] CSP saved to {csp_path}")
    
    def load(self, model_path=None, csp_path=None):
        """
        Load a previously trained model from disk.
        
        Args:
            model_path: Path to LDA model. Default from config.
            csp_path: Path to CSP filters. Default from config.
        """
        model_path = model_path or config.MODEL_FILE
        csp_path = csp_path or config.CSP_FILE
        
        self.lda = joblib.load(model_path)
        self.csp = joblib.load(csp_path)
        self.is_trained = True
        
        print(f"[Classifier] Model loaded from {model_path}")
