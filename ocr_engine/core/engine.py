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

from .result import OCRResult, TextBox, PDFResult
from ..preprocessing.preprocessor import ImagePreprocessor
from ..utils.image_utils import load_image, validate_image_format, is_pdf_file, pdf_to_images

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
                # 只使用最基本的参数，避免版本兼容性问题
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang
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
            # 尝试使用 cls 参数，如果不支持则不传入
            try:
                result = self.ocr.ocr(img_array, cls=True)
            except TypeError:
                result = self.ocr.ocr(img_array)
        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            raise RuntimeError(f"OCR recognition failed: {e}")
        
        processing_time = time.time() - start_time
        
        # 解析结果
        text_boxes = []
        full_text_parts = []
        
        if result:
            # 处理不同版本的返回格式
            if isinstance(result, list) and len(result) > 0:
                # 新版 PaddleOCR 返回格式: [[line1, line2, ...]]
                lines = result[0] if isinstance(result[0], list) else result
                for line in lines:
                    if line and len(line) >= 2:
                        try:
                            box = line[0]
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                            else:
                                text = str(text_info)
                                confidence = 0.0
                            
                            text_boxes.append(TextBox(
                                text=text,
                                confidence=float(confidence) if confidence else 0.0,
                                box=[[int(p[0]), int(p[1])] for p in box] if box else []
                            ))
                            full_text_parts.append(text)
                        except Exception as parse_error:
                            logger.warning(f"解析行失败: {parse_error}, line: {line}")
                            continue
        
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
        try:
            result = self.ocr.ocr(img_array, cls=False, rec=False)
        except TypeError:
            result = self.ocr.ocr(img_array)
        
        regions = []
        if result:
            # 处理不同版本的返回格式
            if isinstance(result, list) and len(result) > 0:
                lines = result[0] if isinstance(result[0], list) else result
                for line in lines:
                    if line and len(line) >= 2:
                        try:
                            box = line[0]
                            confidence = line[1] if len(line) > 1 else 0.0
                            regions.append({
                                "box": [[int(p[0]), int(p[1])] for p in box] if box else [],
                                "confidence": float(confidence) if confidence else 0.0
                            })
                        except Exception as parse_error:
                            logger.warning(f"解析区域失败: {parse_error}, line: {line}")
                            continue
        
        return regions
    
    def recognize_pdf(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 200,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
        return_details: bool = True
    ) -> PDFResult:
        """
        识别PDF文件中的文字
        
        Args:
            pdf_path: PDF文件路径
            dpi: 转换分辨率，默认200
            first_page: 起始页码（从1开始）
            last_page: 结束页码
            return_details: 是否返回详细信息
            
        Returns:
            PDFResult: PDF识别结果
        """
        from ..utils.image_utils import get_pdf_info
        
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        # 检查是否为PDF文件
        if not is_pdf_file(pdf_path):
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        # 获取PDF信息
        pdf_info = get_pdf_info(pdf_path)
        logger.info(f"Processing PDF: {pdf_path.name}, {pdf_info['page_count']} pages")
        
        # 将PDF转换为图像
        page_images = pdf_to_images(pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)
        
        # 识别每一页
        page_results = []
        all_text_parts = []
        
        for page_num, img_array in page_images:
            logger.info(f"Processing page {page_num}/{pdf_info['page_count']}...")
            
            # 识别当前页
            result = self.recognize(img_array, return_details=return_details)
            result.metadata["page_number"] = page_num
            
            page_results.append(result)
            if result.text:
                # 清理文本中的页面标识符（如 -1-、- 8 - 等）
                import re
                cleaned_text = result.text
                # 移除类似 "- 1 -"、"-1-"、"- 8 -"、"-8 -" 的页面标识
                cleaned_text = re.sub(r'\n?\s*-\s*\d+\s*-\s*\n?', '\n', cleaned_text)
                cleaned_text = re.sub(r'\n?\s*-\d+-\s*\n?', '\n', cleaned_text)
                cleaned_text = re.sub(r'\n?\s*-\d+\s+-\s*\n?', '\n', cleaned_text)
                # 移除行尾的页面标识（如 "文本内容\n- 8 -"）
                cleaned_text = re.sub(r'\n+\s*-\s*\d+\s*-\s*$', '', cleaned_text)
                cleaned_text = re.sub(r'\n+\s*-\d+-\s*$', '', cleaned_text)
                # 移除多余的空行
                cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
                all_text_parts.append(cleaned_text.strip())
        
        total_time = time.time() - start_time
        full_text = "\n\n".join(all_text_parts)
        
        return PDFResult(
            text=full_text,
            page_results=page_results,
            total_pages=pdf_info['page_count'],
            processed_pages=len(page_results),
            processing_time=total_time,
            pdf_path=str(pdf_path),
            metadata={
                "language": self.lang,
                "dpi": dpi,
                "pdf_metadata": pdf_info['metadata']
            }
        )
    
    def get_engine_info(self) -> Dict[str, Any]:
        """获取引擎信息"""
        return {
            "version": "1.0.0",
            "language": self.lang,
            "use_gpu": self.use_gpu,
            "enable_preprocess": self.enable_preprocess,
            "preprocess_config": self.preprocess_config,
            "supports_pdf": True
        }
