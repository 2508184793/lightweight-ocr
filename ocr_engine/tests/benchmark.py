"""
性能测试模块
评估OCR系统的识别准确率、处理速度等关键指标
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.engine import OCREngine
from ..utils.image_utils import batch_load_images, get_image_info


@dataclass
class BenchmarkResult:
    """性能测试结果"""
    total_images: int
    successful: int
    failed: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    avg_confidence: float
    throughput: float  # images per second
    details: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def calculate_cer(predicted: str, ground_truth: str) -> float:
    """
    计算字符错误率(Character Error Rate, CER)
    
    Args:
        predicted: 预测文本
        ground_truth: 真实文本
        
    Returns:
        CER值 (0-1之间，越小越好)
    """
    if not ground_truth:
        return 1.0 if predicted else 0.0
    
    # 使用Levenshtein距离
    m, n = len(ground_truth), len(predicted)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ground_truth[i - 1] == predicted[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    
    return dp[m][n] / max(m, n)


def calculate_wer(predicted: str, ground_truth: str) -> float:
    """
    计算词错误率(Word Error Rate, WER)
    
    Args:
        predicted: 预测文本
        ground_truth: 真实文本
        
    Returns:
        WER值 (0-1之间，越小越好)
    """
    pred_words = predicted.split()
    truth_words = ground_truth.split()
    
    if not truth_words:
        return 1.0 if pred_words else 0.0
    
    m, n = len(truth_words), len(pred_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if truth_words[i - 1] == pred_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    
    return dp[m][n] / max(m, n)


def run_benchmark(
    test_data_path: str,
    ground_truth_file: Optional[str] = None,
    lang: str = "ch",
    enable_preprocess: bool = True,
    warmup: int = 3
) -> Dict[str, Any]:
    """
    运行性能测试
    
    Args:
        test_data_path: 测试图片文件夹路径
        ground_truth_file: 真实标签文件路径（可选，JSON格式）
        lang: 识别语言
        enable_preprocess: 是否启用预处理
        warmup: 预热次数
        
    Returns:
        测试结果字典
    """
    print(f"开始性能测试...")
    print(f"测试数据: {test_data_path}")
    print(f"语言: {lang}")
    print(f"预处理: {enable_preprocess}")
    
    # 加载测试数据
    images = batch_load_images(test_data_path)
    if not images:
        raise ValueError(f"未找到测试图片: {test_data_path}")
    
    print(f"找到 {len(images)} 张测试图片")
    
    # 加载真实标签（如果有）
    ground_truths = {}
    if ground_truth_file and Path(ground_truth_file).exists():
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truths = json.load(f)
    
    # 初始化引擎
    engine = OCREngine(lang=lang, enable_preprocess=enable_preprocess)
    
    # 预热
    print(f"预热中... ({warmup}次)")
    for i in range(min(warmup, len(images))):
        engine.recognize(images[i][1])
    
    # 正式测试
    print("开始正式测试...")
    results = []
    processing_times = []
    confidences = []
    
    for i, (img_path, img_array) in enumerate(images):
        try:
            start_time = time.time()
            result = engine.recognize(img_array)
            process_time = time.time() - start_time
            
            processing_times.append(process_time)
            
            # 收集置信度
            if result.text_boxes:
                avg_conf = sum(box.confidence for box in result.text_boxes) / len(result.text_boxes)
                confidences.append(avg_conf)
            
            # 计算准确率（如果有真实标签）
            cer = None
            wer = None
            img_name = Path(img_path).name
            if img_name in ground_truths:
                cer = calculate_cer(result.text, ground_truths[img_name])
                wer = calculate_wer(result.text, ground_truths[img_name])
            
            results.append({
                "file": img_path,
                "text": result.text,
                "processing_time": process_time,
                "confidence": confidences[-1] if confidences else 0,
                "text_boxes_count": len(result.text_boxes),
                "cer": cer,
                "wer": wer,
                "success": True
            })
            
            print(f"  [{i+1}/{len(images)}] {img_path} - {process_time:.3f}s")
            
        except Exception as e:
            results.append({
                "file": img_path,
                "error": str(e),
                "success": False
            })
            print(f"  [{i+1}/{len(images)}] {img_path} - 失败: {e}")
    
    # 计算统计信息
    successful_results = [r for r in results if r.get("success", False)]
    failed_count = len(results) - len(successful_results)
    
    if processing_times:
        total_time = sum(processing_times)
        avg_time = statistics.mean(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        throughput = len(successful_results) / total_time if total_time > 0 else 0
    else:
        total_time = avg_time = min_time = max_time = throughput = 0
    
    avg_confidence = statistics.mean(confidences) if confidences else 0
    
    # 计算平均CER和WER（如果有真实标签）
    cers = [r["cer"] for r in results if r.get("cer") is not None]
    wers = [r["wer"] for r in results if r.get("wer") is not None]
    avg_cer = statistics.mean(cers) if cers else None
    avg_wer = statistics.mean(wers) if wers else None
    
    benchmark_result = {
        "summary": {
            "total_images": len(images),
            "successful": len(successful_results),
            "failed": failed_count,
            "total_time": round(total_time, 3),
            "avg_time": round(avg_time, 3),
            "min_time": round(min_time, 3),
            "max_time": round(max_time, 3),
            "avg_confidence": round(avg_confidence, 4),
            "throughput": round(throughput, 2),
            "avg_cer": round(avg_cer, 4) if avg_cer is not None else None,
            "avg_wer": round(avg_wer, 4) if avg_wer is not None else None
        },
        "configuration": {
            "language": lang,
            "enable_preprocess": enable_preprocess,
            "engine_info": engine.get_engine_info()
        },
        "details": results
    }
    
    # 打印摘要
    print("\n" + "="*50)
    print("性能测试摘要")
    print("="*50)
    print(f"总图片数: {len(images)}")
    print(f"成功: {len(successful_results)}")
    print(f"失败: {failed_count}")
    print(f"总时间: {total_time:.3f}s")
    print(f"平均时间: {avg_time:.3f}s")
    print(f"最短时间: {min_time:.3f}s")
    print(f"最长时间: {max_time:.3f}s")
    print(f"吞吐量: {throughput:.2f} 张/秒")
    print(f"平均置信度: {avg_confidence:.4f}")
    if avg_cer is not None:
        print(f"平均CER: {avg_cer:.4f}")
        print(f"平均WER: {avg_wer:.4f}")
    print("="*50)
    
    return benchmark_result


def generate_test_image(
    text: str,
    output_path: str,
    size: Tuple[int, int] = (800, 200),
    font_size: int = 32,
    add_noise: bool = False,
    rotation: float = 0
) -> None:
    """
    生成测试图片（用于测试）
    
    Args:
        text: 图片中的文字
        output_path: 输出路径
        size: 图片尺寸
        font_size: 字体大小
        add_noise: 是否添加噪声
        rotation: 旋转角度
    """
    # 创建白色背景
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 绘制文字
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill='black', font=font)
    
    # 旋转
    if rotation != 0:
        image = image.rotate(rotation, fillcolor='white')
    
    # 添加噪声
    if add_noise:
        img_array = np.array(image)
        noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def create_test_dataset(
    output_dir: str,
    num_samples: int = 10,
    texts: Optional[List[str]] = None
) -> str:
    """
    创建测试数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        texts: 自定义文本列表
        
    Returns:
        数据集路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    default_texts = [
        "Hello World",
        "这是一段中文测试文本",
        "OCR Engine Test 123",
        "文字识别技术",
        "Lightweight OCR System",
        "深度学习与自然语言处理",
        "Python Programming",
        "人工智能与计算机视觉",
        "Machine Learning",
        "百度飞桨PaddleOCR"
    ]
    
    texts = texts or default_texts
    ground_truths = {}
    
    for i in range(min(num_samples, len(texts))):
        text = texts[i]
        img_path = output_path / f"test_{i:03d}.png"
        
        generate_test_image(text, str(img_path))
        ground_truths[img_path.name] = text
    
    # 保存真实标签
    gt_path = output_path / "ground_truth.json"
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truths, f, ensure_ascii=False, indent=2)
    
    print(f"测试数据集已创建: {output_path}")
    print(f"包含 {len(ground_truths)} 张测试图片")
    
    return str(output_path)


if __name__ == "__main__":
    # 创建测试数据集并运行测试
    test_dir = "test_data"
    create_test_dataset(test_dir, num_samples=5)
    
    results = run_benchmark(test_dir)
    
    # 保存结果
    with open("benchmark_result.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 benchmark_result.json")
