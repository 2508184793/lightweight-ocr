# Lightweight OCR Engine

基于非大模型技术的高性能OCR系统，采用PaddleOCR作为核心引擎，结合自定义图像预处理模块，提供工业级文字识别能力。

## 特性

- 🚀 **高性能**：基于PP-OCRv4，中文识别准确率>95%
- 🎯 **轻量级**：模型仅8.1MB，适合各种部署环境
- 🔧 **易集成**：简洁的API设计，支持多种调用方式
- 🖼️ **强预处理**：内置去噪、二值化、倾斜校正等功能
- 📊 **可测试**：包含完整的性能测试和评估模块
- 🌐 **多格式**：支持JPG、PNG、TIFF、BMP等常见格式

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础使用

```python
from ocr_engine import OCREngine

# 创建OCR引擎实例
engine = OCREngine()

# 识别图片
result = engine.recognize("path/to/image.jpg")
print(result.text)
```

### 命令行使用

```bash
# 识别单张图片
python -m ocr_engine recognize image.jpg

# 批量识别
python -m ocr_engine recognize folder/ --output results.json

# 性能测试
python -m ocr_engine benchmark --test-data test_images/
```

## 项目结构

```
ocr_engine/
├── core/           # 核心OCR引擎
├── preprocessing/  # 图像预处理模块
├── utils/          # 工具函数
├── api/            # API接口
└── tests/          # 测试模块
```

## 技术栈

- **OCR引擎**：PaddleOCR (PP-OCRv4)
- **图像处理**：OpenCV + Pillow
- **API框架**：FastAPI
- **测试框架**：pytest

## 许可证

MIT License
