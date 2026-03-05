"""
OCR引擎基础使用示例
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_engine import OCREngine
from ocr_engine.tests.benchmark import create_test_dataset


def example_basic_recognition():
    """基础识别示例"""
    print("=" * 50)
    print("示例1: 基础识别")
    print("=" * 50)
    
    # 创建测试图片
    test_dir = Path("temp_test")
    test_dir.mkdir(exist_ok=True)
    
    from PIL import Image, ImageDraw, ImageFont
    
    # 创建测试图片
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 30), "Hello OCR World! 你好世界", fill='black', font=font)
    img_path = test_dir / "test_image.png"
    img.save(img_path)
    
    print(f"创建测试图片: {img_path}")
    
    # 创建OCR引擎
    print("初始化OCR引擎...")
    engine = OCREngine(lang='ch')
    
    # 识别图片
    print(f"识别图片...")
    result = engine.recognize(str(img_path))
    
    # 输出结果
    print(f"\n识别结果:")
    print(f"  文本: {result.text}")
    print(f"  处理时间: {result.processing_time:.3f}秒")
    print(f"  文本框数量: {len(result.text_boxes)}")
    
    if result.text_boxes:
        print(f"\n文本框详情:")
        for i, box in enumerate(result.text_boxes):
            print(f"  [{i+1}] {box.text} (置信度: {box.confidence:.3f})")
    
    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print(f"\n清理临时文件")


def example_with_preprocessing():
    """带预处理的识别示例"""
    print("\n" + "=" * 50)
    print("示例2: 带预处理的识别")
    print("=" * 50)
    
    # 配置预处理
    preprocess_config = {
        "denoise_method": "gaussian",
        "binarize_method": "otsu",
        "enable_deskew": True,
        "enable_contrast_enhance": True,
    }
    
    print("预处理配置:")
    for key, value in preprocess_config.items():
        print(f"  {key}: {value}")
    
    # 创建引擎
    engine = OCREngine(
        lang='ch',
        enable_preprocess=True,
        preprocess_config=preprocess_config
    )
    
    print(f"\n引擎信息: {engine.get_engine_info()}")


def example_batch_recognition():
    """批量识别示例"""
    print("\n" + "=" * 50)
    print("示例3: 批量识别")
    print("=" * 50)
    
    # 创建测试数据集
    print("创建测试数据集...")
    test_dir = create_test_dataset("temp_batch_test", num_samples=3)
    
    # 创建引擎
    engine = OCREngine(lang='ch')
    
    # 批量识别
    print("\n开始批量识别...")
    from ocr_engine.utils.image_utils import batch_load_images
    
    images = batch_load_images(test_dir)
    results = engine.recognize_batch([img for _, img in images])
    
    print(f"\n批量识别结果:")
    for (img_path, _), result in zip(images, results):
        print(f"  {Path(img_path).name}: {result.text[:30]}... ({result.processing_time:.3f}s)")
    
    # 统计
    total_time = sum(r.processing_time for r in results)
    print(f"\n统计:")
    print(f"  总图片数: {len(results)}")
    print(f"  总时间: {total_time:.3f}秒")
    print(f"  平均时间: {total_time/len(results):.3f}秒")
    
    # 清理
    import shutil
    shutil.rmtree("temp_batch_test")


def example_api_usage():
    """API使用示例"""
    print("\n" + "=" * 50)
    print("示例4: API服务")
    print("=" * 50)
    
    print("""
启动API服务:
  uvicorn ocr_engine.api.server:app --host 0.0.0.0 --port 8000

API端点:
  GET  /health          - 健康检查
  GET  /info            - 引擎信息
  POST /recognize       - 识别图片
  POST /recognize/base64 - Base64图片识别
  POST /detect          - 文本区域检测

使用示例:
  curl -X POST "http://localhost:8000/recognize" \\
    -F "file=@image.jpg" \\
    -F "language=ch"
""")


if __name__ == "__main__":
    print("OCR引擎使用示例")
    print("=" * 50)
    
    try:
        example_basic_recognition()
        example_with_preprocessing()
        example_batch_recognition()
        example_api_usage()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
