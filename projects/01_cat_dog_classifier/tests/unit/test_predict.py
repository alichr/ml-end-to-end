"""Unit tests for prediction/preprocessing logic."""

import numpy as np
from PIL import Image

from src.serving.predict import _preprocess_image, _softmax


class TestPreprocessImage:
    def test_output_shape(self, sample_image: Image.Image) -> None:
        result = _preprocess_image(sample_image)
        assert result.shape == (1, 3, 224, 224)

    def test_output_dtype(self, sample_image: Image.Image) -> None:
        result = _preprocess_image(sample_image)
        assert result.dtype == np.float32

    def test_handles_grayscale(self) -> None:
        gray = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode="L")
        result = _preprocess_image(gray)
        assert result.shape == (1, 3, 224, 224)

    def test_handles_rgba(self) -> None:
        rgba = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8), mode="RGBA"
        )
        result = _preprocess_image(rgba)
        assert result.shape == (1, 3, 224, 224)

    def test_handles_small_image(self) -> None:
        tiny = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        result = _preprocess_image(tiny)
        assert result.shape == (1, 3, 224, 224)

    def test_handles_large_image(self) -> None:
        large = Image.fromarray(np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8))
        result = _preprocess_image(large)
        assert result.shape == (1, 3, 224, 224)

    def test_deterministic(self, sample_image: Image.Image) -> None:
        r1 = _preprocess_image(sample_image)
        r2 = _preprocess_image(sample_image)
        np.testing.assert_array_equal(r1, r2)


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        logits = np.array([[2.0, 1.0]])
        probs = _softmax(logits)
        np.testing.assert_almost_equal(probs.sum(), 1.0)

    def test_correct_shape(self) -> None:
        logits = np.array([[1.0, 2.0], [3.0, 0.5]])
        probs = _softmax(logits)
        assert probs.shape == (2, 2)

    def test_highest_logit_gets_highest_prob(self) -> None:
        logits = np.array([[0.1, 5.0]])
        probs = _softmax(logits)
        assert probs[0, 1] > probs[0, 0]

    def test_handles_large_values(self) -> None:
        logits = np.array([[1000.0, 1001.0]])
        probs = _softmax(logits)
        assert not np.any(np.isnan(probs))
        np.testing.assert_almost_equal(probs.sum(), 1.0)
