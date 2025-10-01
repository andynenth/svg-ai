"""
Statistical Fallback Classifier - Day 2 Task 4
Feature-based sklearn classifier for logo type classification.
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import time

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class StatisticalClassifier:
    """Feature-based statistical classifier for logo type classification."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize statistical classifier.

        Args:
            model_path: Path to saved model pickle file
        """
        self.class_names = ['simple', 'text', 'gradient', 'complex']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.model = None
        self.feature_scaler = None
        self.is_trained = False

        # Default model path
        if model_path is None:
            models_dir = Path(__file__).parent.parent / "models" / "trained"
            model_path = models_dir / "statistical_classifier.pkl"

        self.model_path = model_path

        # Try to load existing model
        self.load_model()

        # If no model exists, train one
        if not self.is_trained:
            logger.info("No trained model found, training new statistical classifier...")
            self.train_model()

        logger.info(f"Statistical classifier initialized. Trained: {self.is_trained}")

    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract statistical features from image.

        Args:
            image_path: Path to image file

        Returns:
            Feature vector or None if extraction failed
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path).convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Resize to standard size for consistent features
            image = cv2.resize(image, (256, 256))

            features = []

            # 1. Color Histogram Features
            color_features = self._extract_color_histogram(image)
            features.extend(color_features)

            # 2. Edge Density Features
            edge_features = self._extract_edge_features(image)
            features.extend(edge_features)

            # 3. Complexity Features
            complexity_features = self._extract_complexity_features(image)
            features.extend(complexity_features)

            # 4. Shape Features
            shape_features = self._extract_shape_features(image)
            features.extend(shape_features)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None

    def _extract_color_histogram(self, image: np.ndarray) -> List[float]:
        """Extract color histogram features."""
        features = []

        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # RGB histograms (8 bins per channel)
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [8], [0, 256])
            features.extend(hist.flatten() / hist.sum())

        # HSV histograms (8 bins per channel)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [8], [0, 256] if i > 0 else [0, 180])
            features.extend(hist.flatten() / hist.sum())

        # Color diversity metrics
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        features.append(unique_colors / (256 * 256 * 256))

        # Dominant color analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.append(np.mean(gray) / 255.0)  # Brightness
        features.append(np.std(gray) / 255.0)   # Contrast

        return features

    def _extract_edge_features(self, image: np.ndarray) -> List[float]:
        """Extract edge density and pattern features."""
        features = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)

        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

        features.append(np.mean(sobel_magnitude) / 255.0)
        features.append(np.std(sobel_magnitude) / 255.0)

        # Laplacian (second derivative)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(np.mean(np.abs(laplacian)) / 255.0)

        # Edge orientation analysis
        orientations = np.arctan2(sobely, sobelx)
        hist, _ = np.histogram(orientations, bins=8, range=(-np.pi, np.pi))
        features.extend(hist / hist.sum())

        return features

    def _extract_complexity_features(self, image: np.ndarray) -> List[float]:
        """Extract complexity and texture features."""
        features = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Local Binary Pattern (simplified)
        lbp = self._calculate_lbp(gray)
        features.append(np.mean(lbp) / 255.0)
        features.append(np.std(lbp) / 255.0)

        # Gradient magnitude statistics
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.append(np.mean(magnitude) / 255.0)
        features.append(np.percentile(magnitude, 90) / 255.0)

        # Entropy (measure of randomness)
        entropy = self._calculate_entropy(gray)
        features.append(entropy)

        # Fractal dimension (simplified)
        fractal_dim = self._calculate_fractal_dimension(gray)
        features.append(fractal_dim)

        return features

    def _extract_shape_features(self, image: np.ndarray) -> List[float]:
        """Extract shape and geometric features."""
        features = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find contours
        contours, _ = cv2.findContours(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Contour properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            if perimeter > 0:
                features.append(4 * np.pi * area / (perimeter**2))  # Circularity
            else:
                features.append(0)

            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            features.append(w / h if h > 0 else 1.0)

            # Extent (contour area / bounding box area)
            extent = area / (w * h) if w * h > 0 else 0
            features.append(extent)

            # Solidity (contour area / convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            features.append(solidity)
        else:
            features.extend([0, 1, 0, 0])  # Default values

        # Number of contours (complexity indicator)
        features.append(min(len(contours), 100) / 100.0)

        return features

    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate simplified Local Binary Pattern."""
        rows, cols = image.shape
        lbp = np.zeros_like(image)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = image[i, j]
                pattern = 0
                pattern |= (image[i-1, j-1] >= center) << 7
                pattern |= (image[i-1, j] >= center) << 6
                pattern |= (image[i-1, j+1] >= center) << 5
                pattern |= (image[i, j+1] >= center) << 4
                pattern |= (image[i+1, j+1] >= center) << 3
                pattern |= (image[i+1, j] >= center) << 2
                pattern |= (image[i+1, j-1] >= center) << 1
                pattern |= (image[i, j-1] >= center) << 0
                lbp[i, j] = pattern

        return lbp

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy."""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy / 8.0  # Normalize

    def _calculate_fractal_dimension(self, image: np.ndarray) -> float:
        """Calculate simplified fractal dimension."""
        try:
            # Box counting method (simplified)
            p = np.where(image < 128, 1, 0)  # Binary image
            scales = [2, 4, 8, 16]
            counts = []

            for scale in scales:
                h, w = p.shape
                new_h, new_w = h // scale, w // scale
                if new_h == 0 or new_w == 0:
                    continue

                # Downsample
                downsampled = p[:new_h*scale, :new_w*scale].reshape(new_h, scale, new_w, scale)
                boxes = np.any(downsampled, axis=(1, 3))
                counts.append(np.sum(boxes))

            if len(counts) >= 2:
                # Linear regression to find slope
                log_scales = np.log(scales[:len(counts)])
                log_counts = np.log(np.array(counts) + 1)
                slope = np.polyfit(log_scales, log_counts, 1)[0]
                return min(max(-slope, 0), 2)  # Clamp between 0 and 2
            else:
                return 1.0
        except:
            return 1.0

    def train_model(self) -> bool:
        """
        Train the statistical classifier.

        Returns:
            True if training successful, False otherwise
        """
        try:
            # Load training data
            X, y = self._load_training_data()

            if len(X) == 0:
                logger.error("No training data available")
                return False

            logger.info(f"Training with {len(X)} samples across {len(set(y))} classes")

            # Split data
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                # If very few samples, use all for training
                X_train, X_test, y_train, y_test = X, X, y, y

            # Scale features
            from sklearn.preprocessing import StandardScaler
            self.feature_scaler = StandardScaler()
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)

            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            )

            start_time = time.time()
            self.model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time

            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Training accuracy: {accuracy:.3f}")

            # Detailed report if enough samples
            if len(X) > 20:
                report = classification_report(y_test, y_pred, target_names=self.class_names)
                logger.info(f"Classification report:\n{report}")

            self.is_trained = True

            # Save model
            self.save_model()

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def _load_training_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load training data from categorized images."""
        X, y = [], []

        # Load from categorized data
        data_dir = PROJECT_ROOT / "data" / "logos"

        for class_idx, class_name in enumerate(self.class_names):
            # Map class names to actual directory names
            if class_name == 'simple':
                class_dir = data_dir / "simple_geometric"
            elif class_name == 'text':
                class_dir = data_dir / "text_based"
            elif class_name == 'gradient':
                class_dir = data_dir / "gradients"
            elif class_name == 'complex':
                class_dir = data_dir / "complex"
            else:
                class_dir = data_dir / class_name

            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            image_files = list(class_dir.glob("*.png"))
            logger.info(f"Loading {len(image_files)} images for class '{class_name}'")

            for image_file in image_files:
                features = self.extract_features(str(image_file))
                if features is not None:
                    X.append(features)
                    y.append(class_idx)

        logger.info(f"Loaded {len(X)} training samples")
        return X, y

    def classify(self, image_path: str) -> Dict[str, Any]:
        """
        Classify logo type from image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with classification results
        """
        start_time = time.time()

        try:
            # Check if model is trained
            if not self.is_trained or self.model is None:
                return self._create_error_result(
                    "Statistical classifier not trained", image_path
                )

            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                return self._create_error_result(
                    "Feature extraction failed", image_path
                )

            # Scale features
            if self.feature_scaler is not None:
                features = features.reshape(1, -1)
                features = self.feature_scaler.transform(features)
            else:
                features = features.reshape(1, -1)

            # Predict
            probabilities = self.model.predict_proba(features)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]

            logo_type = self.class_names[predicted_class]

            # Get all class probabilities
            all_probs = {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            }

            processing_time = time.time() - start_time

            return {
                'logo_type': logo_type,
                'confidence': float(confidence),
                'all_probabilities': all_probs,
                'model_type': 'statistical_classifier',
                'processing_time': processing_time,
                'image_path': image_path,
                'success': True
            }

        except Exception as e:
            logger.error(f"Classification failed for {image_path}: {e}")
            return self._create_error_result(str(e), image_path)

    def _create_error_result(self, error_message: str, image_path: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'logo_type': 'unknown',
            'confidence': 0.0,
            'all_probabilities': {name: 0.0 for name in self.class_names},
            'model_type': 'statistical_classifier',
            'processing_time': 0.0,
            'image_path': image_path,
            'error': error_message,
            'success': False
        }

    def save_model(self) -> bool:
        """Save trained model to pickle file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            model_data = {
                'model': self.model,
                'feature_scaler': self.feature_scaler,
                'class_names': self.class_names,
                'is_trained': self.is_trained
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self) -> bool:
        """Load trained model from pickle file."""
        try:
            if not os.path.exists(self.model_path):
                logger.info(f"Model file not found: {self.model_path}")
                return False

            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.feature_scaler = model_data['feature_scaler']
            self.class_names = model_data['class_names']
            self.is_trained = model_data['is_trained']

            logger.info(f"Model loaded from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the statistical classifier."""
        # Estimate feature count by testing feature extraction
        feature_count = 'unknown'
        try:
            # Try to extract features from a dummy image to count them
            test_features = np.zeros((64, 64, 3), dtype=np.uint8)
            import tempfile
            import cv2
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, test_features)
                test_result = self.extract_features(tmp.name)
                if test_result is not None:
                    feature_count = len(test_result)
                os.unlink(tmp.name)
        except:
            feature_count = 'variable'

        return {
            'model_name': 'Statistical Classifier (Random Forest)',
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'is_trained': self.is_trained,
            'model_path': str(self.model_path),
            'feature_count': feature_count,
            'algorithm': 'Random Forest + Feature Engineering'
        }