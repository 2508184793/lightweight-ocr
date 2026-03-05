"""
图像预处理模块
提供去噪、二值化、倾斜校正等功能
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from enum import Enum


class DenoiseMethod(Enum):
    """去噪方法枚举"""
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"
    NONE = "none"


class BinarizeMethod(Enum):
    """二值化方法枚举"""
    OTSU = "otsu"
    ADAPTIVE = "adaptive"
    SAUVOLA = "sauvola"
    NONE = "none"


class ImagePreprocessor:
    """
    图像预处理器
    
    提供完整的OCR前处理流程，包括：
    - 图像去噪（高斯滤波、中值滤波、双边滤波）
    - 图像二值化（Otsu、自适应阈值、Sauvola）
    - 倾斜校正（霍夫变换、投影法）
    - 图像增强（对比度、锐化）
    """
    
    def __init__(
        self,
        denoise_method: str = "gaussian",
        binarize_method: str = "otsu",
        enable_deskew: bool = True,
        enable_contrast_enhance: bool = True,
        denoise_kernel_size: int = 5,
        binarize_block_size: int = 11,
        contrast_clip_limit: float = 2.0,
        contrast_grid_size: Tuple[int, int] = (8, 8)
    ):
        """
        初始化预处理器
        
        Args:
            denoise_method: 去噪方法 ('gaussian', 'median', 'bilateral', 'none')
            binarize_method: 二值化方法 ('otsu', 'adaptive', 'sauvola', 'none')
            enable_deskew: 是否启用倾斜校正
            enable_contrast_enhance: 是否启用对比度增强
            denoise_kernel_size: 去噪核大小
            binarize_block_size: 自适应二值化块大小
            contrast_clip_limit: CLAHE对比度限制
            contrast_grid_size: CLAHE网格大小
        """
        self.denoise_method = DenoiseMethod(denoise_method)
        self.binarize_method = BinarizeMethod(binarize_method)
        self.enable_deskew = enable_deskew
        self.enable_contrast_enhance = enable_contrast_enhance
        self.denoise_kernel_size = denoise_kernel_size
        self.binarize_block_size = binarize_block_size
        self.contrast_clip_limit = contrast_clip_limit
        self.contrast_grid_size = contrast_grid_size
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        执行完整的预处理流程
        
        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            
        Returns:
            预处理后的图像
        """
        result = image.copy()
        
        # 1. 转换为灰度图（如果需要）
        if len(result.shape) == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = result
        
        # 2. 对比度增强
        if self.enable_contrast_enhance:
            gray = self._enhance_contrast(gray)
        
        # 3. 去噪
        if self.denoise_method != DenoiseMethod.NONE:
            gray = self._denoise(gray)
        
        # 4. 二值化
        if self.binarize_method != BinarizeMethod.NONE:
            binary = self._binarize(gray)
        else:
            binary = gray
        
        # 5. 倾斜校正
        if self.enable_deskew:
            binary = self._deskew(binary)
        
        return binary
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image: 灰度图像
            
        Returns:
            去噪后的图像
        """
        if self.denoise_method == DenoiseMethod.GAUSSIAN:
            return cv2.GaussianBlur(
                image, 
                (self.denoise_kernel_size, self.denoise_kernel_size), 
                0
            )
        
        elif self.denoise_method == DenoiseMethod.MEDIAN:
            return cv2.medianBlur(image, self.denoise_kernel_size)
        
        elif self.denoise_method == DenoiseMethod.BILATERAL:
            return cv2.bilateralFilter(
                image, 
                self.denoise_kernel_size, 
                self.denoise_kernel_size * 2, 
                self.denoise_kernel_size / 2
            )
        
        return image
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        图像二值化
        
        Args:
            image: 灰度图像
            
        Returns:
            二值化图像
        """
        if self.binarize_method == BinarizeMethod.OTSU:
            # Otsu自动阈值
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        
        elif self.binarize_method == BinarizeMethod.ADAPTIVE:
            # 自适应阈值
            return cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.binarize_block_size,
                2
            )
        
        elif self.binarize_method == BinarizeMethod.SAUVOLA:
            # Sauvola阈值（适合光照不均的文档）
            return self._sauvola_threshold(image)
        
        return image
    
    def _sauvola_threshold(self, image: np.ndarray, window_size: int = 15, k: float = 0.2, r: float = 128) -> np.ndarray:
        """
        Sauvola局部阈值算法
        适合处理光照不均的文档图像
        
        Args:
            image: 灰度图像
            window_size: 窗口大小
            k: 参数k
            r: 参数R（通常取128）
            
        Returns:
            二值化图像
        """
        # 计算局部均值和标准差
        mean = cv2.blur(image.astype(np.float32), (window_size, window_size))
        mean_sq = cv2.blur(image.astype(np.float32) ** 2, (window_size, window_size))
        std = np.sqrt(mean_sq - mean ** 2)
        
        # Sauvola阈值公式
        threshold = mean * (1 + k * (std / r - 1))
        
        binary = np.where(image > threshold, 255, 0).astype(np.uint8)
        return binary
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        倾斜校正
        使用霍夫变换检测文本行角度并旋转校正
        
        Args:
            image: 二值或灰度图像
            
        Returns:
            校正后的图像
        """
        # 检测边缘
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return image
        
        # 计算角度
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # 只考虑接近水平的角度（-45到45度）
                if -45 < angle < 45:
                    angles.append(angle)
        
        if not angles:
            return image
        
        # 使用中位数角度作为旋转角度
        median_angle = np.median(angles)
        
        # 如果角度很小，不需要旋转
        if abs(median_angle) < 0.5:
            return image
        
        # 旋转图像
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        return rotated
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        对比度增强（CLAHE）
        
        Args:
            image: 灰度图像
            
        Returns:
            增强后的图像
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.contrast_clip_limit,
            tileGridSize=self.contrast_grid_size
        )
        return clahe.apply(image)
    
    def preprocess_for_display(self, image: np.ndarray) -> np.ndarray:
        """
        预处理用于显示（保持3通道）
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的3通道图像
        """
        processed = self.process(image)
        
        # 如果是单通道，转换为3通道
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            "denoise_method": self.denoise_method.value,
            "binarize_method": self.binarize_method.value,
            "enable_deskew": self.enable_deskew,
            "enable_contrast_enhance": self.enable_contrast_enhance,
            "denoise_kernel_size": self.denoise_kernel_size,
            "binarize_block_size": self.binarize_block_size,
            "contrast_clip_limit": self.contrast_clip_limit,
            "contrast_grid_size": self.contrast_grid_size
        }
