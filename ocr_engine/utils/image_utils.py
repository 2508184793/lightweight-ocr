"""
图像工具函数
"""

import os
from pathlib import Path
from typing import Union, Tuple, List, Optional
import numpy as np
from PIL import Image
import cv2


# 支持的图像格式
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
SUPPORTED_PDF_FORMATS = {'.pdf'}
ALL_SUPPORTED_FORMATS = SUPPORTED_FORMATS | SUPPORTED_PDF_FORMATS


def validate_image_format(image_path: Union[str, Path]) -> bool:
    """
    验证图像格式是否支持
    
    Args:
        image_path: 图像路径
        
    Returns:
        bool: 是否支持
        
    Raises:
        ValueError: 格式不支持时抛出
    """
    path = Path(image_path)
    ext = path.suffix.lower()
    
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {ext}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    return True


def load_image(image_path: Union[str, Path], mode: str = "auto") -> np.ndarray:
    """
    加载图像
    
    Args:
        image_path: 图像路径
        mode: 加载模式 ('auto', 'rgb', 'bgr', 'gray')
        
    Returns:
        numpy.ndarray: 图像数组
    """
    validate_image_format(image_path)
    
    if mode == "auto":
        # 使用OpenCV加载，保持原始格式
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    
    elif mode == "rgb":
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    elif mode == "bgr":
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    
    elif mode == "gray":
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    保存图像
    
    Args:
        image: 图像数组
        output_path: 输出路径
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to save image: {output_path}")


def resize_image(
    image: np.ndarray,
    max_size: Tuple[int, int] = (1920, 1080),
    keep_ratio: bool = True
) -> np.ndarray:
    """
    调整图像大小（保持长宽比）
    
    Args:
        image: 输入图像
        max_size: 最大尺寸 (宽, 高)
        keep_ratio: 是否保持长宽比
        
    Returns:
        调整后的图像
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size
    
    if keep_ratio:
        # 计算缩放比例
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w, new_h = max_w, max_h
    
    if (new_w, new_h) != (w, h):
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def get_image_info(image_path: Union[str, Path]) -> dict:
    """
    获取图像信息
    
    Args:
        image_path: 图像路径
        
    Returns:
        图像信息字典
    """
    validate_image_format(image_path)
    
    # 使用PIL获取信息
    with Image.open(image_path) as img:
        info = {
            "path": str(image_path),
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height,
            "file_size": Path(image_path).stat().st_size
        }
    
    return info


def convert_to_format(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    target_format: str = "png"
) -> None:
    """
    转换图像格式
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        target_format: 目标格式
    """
    img = load_image(image_path, mode="rgb")
    
    # 转换回BGR用于OpenCV保存
    if target_format.lower() in ['jpg', 'jpeg']:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    save_image(img, output_path)


def batch_load_images(
    folder_path: Union[str, Path],
    recursive: bool = False
) -> List[Tuple[str, np.ndarray]]:
    """
    批量加载文件夹中的图像
    
    Args:
        folder_path: 文件夹路径
        recursive: 是否递归子文件夹
        
    Returns:
        (文件路径, 图像数组) 列表
    """
    folder = Path(folder_path)
    
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")
    
    pattern = "**/*" if recursive else "*"
    image_files = []
    
    for ext in SUPPORTED_FORMATS:
        image_files.extend(folder.glob(f"{pattern}{ext}"))
        image_files.extend(folder.glob(f"{pattern}{ext.upper()}"))
    
    results = []
    for img_path in sorted(set(image_files)):
        try:
            img = load_image(img_path)
            results.append((str(img_path), img))
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
    
    return results


def draw_text_boxes(
    image: np.ndarray,
    text_boxes: List[dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制文本框
    
    Args:
        image: 输入图像
        text_boxes: 文本框列表，每个包含 'box' 键
        color: 框颜色 (B, G, R)
        thickness: 线宽
        
    Returns:
        绘制后的图像
    """
    result = image.copy()
    
    for box_info in text_boxes:
        box = box_info.get("box", [])
        if len(box) == 4:
            # 绘制四边形
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(result, [pts], True, color, thickness)
    
    return result


def is_pdf_file(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否为PDF格式
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否为PDF文件
    """
    path = Path(file_path)
    return path.suffix.lower() in SUPPORTED_PDF_FORMATS


def pdf_to_images(
    pdf_path: Union[str, Path],
    dpi: int = 200,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None
) -> List[Tuple[int, np.ndarray]]:
    """
    将PDF转换为图像列表
    
    Args:
        pdf_path: PDF文件路径
        dpi: 转换分辨率，默认200
        first_page: 起始页码（从1开始），默认None表示从第一页开始
        last_page: 结束页码，默认None表示到最后一页
        
    Returns:
        List[Tuple[int, np.ndarray]]: [(页码, 图像数组), ...]
        
    Raises:
        ImportError: 未安装pymupdf时抛出
        FileNotFoundError: 文件不存在时抛出
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PDF processing requires PyMuPDF. "
            "Please install with: pip install PyMuPDF"
        )
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # 打开PDF
    doc = fitz.open(str(pdf_path))
    images = []
    
    # 确定页码范围
    start_page = (first_page - 1) if first_page else 0
    end_page = last_page if last_page else len(doc)
    start_page = max(0, start_page)
    end_page = min(len(doc), end_page)
    
    for page_num in range(start_page, end_page):
        page = doc[page_num]
        
        # 将页面转换为图像
        mat = fitz.Matrix(dpi/72, dpi/72)  # 72是PDF默认DPI
        pix = page.get_pixmap(matrix=mat)
        
        # 转换为numpy数组
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        # 如果是RGBA，转换为RGB
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        images.append((page_num + 1, img))
    
    doc.close()
    return images


def get_pdf_info(pdf_path: Union[str, Path]) -> dict:
    """
    获取PDF文件信息
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        PDF信息字典
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PDF processing requires PyMuPDF. Please install with: pip install PyMuPDF")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(str(pdf_path))
    
    info = {
        "path": str(pdf_path),
        "page_count": len(doc),
        "file_size": pdf_path.stat().st_size,
        "metadata": doc.metadata
    }
    
    doc.close()
    return info
