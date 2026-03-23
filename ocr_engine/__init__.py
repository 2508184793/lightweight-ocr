"""
Lightweight OCR Engine
基于非大模型技术的高性能OCR系统
"""

from .core.engine import OCREngine
from .core.result import OCRResult, TextBox, PDFResult
from .preprocessing.preprocessor import ImagePreprocessor

__version__ = "1.0.0"
__all__ = ["OCREngine", "OCRResult", "TextBox", "PDFResult", "ImagePreprocessor"]
