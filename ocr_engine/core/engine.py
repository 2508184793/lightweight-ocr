"""
OCR引擎核心实现
基于PaddleOCR的非大模型OCR解决方案
"""

import time
import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import numpy as np
from PIL import Image

from .result import OCRResult, TextBox
from ..preprocessing.preprocessor import ImagePreprocessor
from ..utils.image_utils import load_image, validate_image_format

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR引擎主类
    
    基于PaddleOCR的轻量级OCR解决方案，支持图像预处理、文本检测和识别
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "ch",
        enable_preprocess: bool = True,
        preprocess_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化OCR引擎
        
        Args:
            use_gpu: 是否使用GPU加速
            lang: 识别语言，支持 'ch'(中文), 'en'(英文), 'ch_en'(中英文) 等
            enable_preprocess: 是否启用图像预处理
            preprocess_config: 预处理配置参数
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.enable_preprocess = enable_preprocess
        self.preprocess_config = preprocess_config or {}
        
        # 初始化预处理器
        self.preprocessor = ImagePreprocessor(**self.preprocess_config) if enable_preprocess else None
        
        # 延迟初始化PaddleOCR（避免导入时加载）
        self._ocr = None
        
        logger.info(f"OCREngine initialized: lang={lang}, use_gpu={use_gpu}, preprocess={enable_preprocess}")
    
    @property
    def ocr(self):
        """延迟初始化PaddleOCR实例"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False
                )
                logger.info("PaddleOCR initialized successfully")
            except ImportError:
                raise ImportError(
                    "PaddleOCR not installed. Please install with: pip install paddleocr"
                )
        return self._ocr
    
    def recognize(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_details: bool = True
    ) -> OCRResult:
        """
        识别图片中的文字
        
        Args:
            image: 图片路径或图片数据
            return_details: 是否返回详细信息（文本框位置等）
            
        Returns:
            OCRResult: 识别结果
        """
        start_time = time.time()
        
        # 加载图片
        if isinstance(image, (str, Path)):
            image_path = str(image)
            validate_image_format(image_path)
            img_array = load_image(image_path)
        elif isinstance(image, Image.Image):
            image_path = None
            img_array = np.array(image)
        else:
            image_path = None
            img_array = image
        
        # 图像预处理
        if self.enable_preprocess and self.preprocessor:
            logger.debug("Applying image preprocessing...")
            img_array = self.preprocessor.process(img_array)
        
        # OCR识别
        try:
            result = self.ocr.ocr(img_array, cls=True)
        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            raise RuntimeError(f"OCR recognition failed: {e}")
        
        processing_time = time.time() - start_time
        
        # 解析结果
        text_boxes = []
        full_text_parts = []
        
        if result and result[0]:
            for line in result[0]:
                if line:
                    box = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    text_boxes.append(TextBox(
                        text=text,
                        confidence=confidence,
                        box=[[int(p[0]), int(p[1])] for p in box]
                    ))
                    full_text_parts.append(text)
        
        full_text = "\n".join(full_text_parts)
        
        return OCRResult(
            text=full_text,
            text_boxes=text_boxes,
            processing_time=processing_time,
            image_path=image_path,
            metadata={
                "language": self.lang,
                "use_gpu": self.use_gpu,
                "preprocessed": self.enable_preprocess
            }
        )
    
    def recognize_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        return_details: bool = True
    ) -> List[OCRResult]:
        """
        批量识别图片
        
        Args:
            images: 图片列表
            return_details: 是否返回详细信息
            
        Returns:
            List[OCRResult]: 识别结果列表
        """
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}...")
            try:
                result = self.recognize(image, return_details)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {e}")
                results.append(OCRResult(
                    text="",
                    text_boxes=[],
                    processing_time=0.0,
                    metadata={"error": str(e)}
                ))
        return results
    
    def detect_text_regions(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        仅检测文本区域（不识别文字）
        
        Args:
            image: 图片路径或图片数据
            
        Returns:
            List[Dict]: 文本区域列表，每个包含box和confidence
        """
        # 加载图片
        if isinstance(image, (str, Path)):
            img_array = load_image(str(image))
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # 预处理
        if self.enable_preprocess and self.preprocessor:
            img_array = self.preprocessor.process(img_array)
        
        # 仅检测
        result = self.ocr.ocr(img_array, cls=False, rec=False)
        
        regions = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    regions.append({
                        "box": [[int(p[0]), int(p[1])] for p in line[0]],
                        "confidence": line[1]
                    })
        
        return regions
    
    def get_engine_info(self) -> Dict[str, Any]:
        """获取引擎信息"""
        return {
            "version": "1.0.0",
            "language": self.lang,
            "use_gpu": self.use_gpu,
            "enable_preprocess": self.enable_preprocess,
            "preprocess_config": self.preprocess_config
        }
