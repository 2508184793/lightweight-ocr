# OCR项目使用说明

## 1. 环境准备

### 1.1 系统要求

- Python 3.8+
- Windows 10/11 或 Linux
- 4GB+ RAM
- 2GB+ 磁盘空间

### 1.2 安装依赖

```bash
# 克隆或下载项目
cd ocr_project

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 1.3 验证安装

```bash
python -c "from ocr_engine import OCREngine; print('安装成功')"
```

## 2. 快速开始

### 2.1 基础使用

```python
from ocr_engine import OCREngine

# 创建OCR引擎（首次使用会自动下载模型）
engine = OCREngine()

# 识别图片
result = engine.recognize('path/to/your/image.jpg')

# 输出结果
print(f"识别文本: {result.text}")
print(f"处理时间: {result.processing_time:.3f}秒")
print(f"文本框数量: {len(result.text_boxes)}")
```

### 2.2 批量识别

```python
from ocr_engine import OCREngine
from pathlib import Path

engine = OCREngine()

# 批量识别文件夹中的图片
image_folder = Path('path/to/images')
images = list(image_folder.glob('*.jpg'))

results = engine.recognize_batch(images)

for img_path, result in zip(images, results):
    print(f"{img_path.name}: {result.text}")
```

## 3. 高级配置

### 3.1 图像预处理配置

```python
from ocr_engine import OCREngine

# 自定义预处理配置
preprocess_config = {
    "denoise_method": "median",        # 去噪方法: gaussian, median, bilateral, none
    "binarize_method": "adaptive",     # 二值化方法: otsu, adaptive, sauvola, none
    "enable_deskew": True,             # 启用倾斜校正
    "enable_contrast_enhance": True,   # 启用对比度增强
    "denoise_kernel_size": 5,          # 去噪核大小
    "binarize_block_size": 11,         # 自适应二值化块大小
}

engine = OCREngine(
    enable_preprocess=True,
    preprocess_config=preprocess_config
)
```

### 3.2 多语言支持

```python
# 中文识别
engine_ch = OCREngine(lang='ch')

# 英文识别
engine_en = OCREngine(lang='en')

# 中英文混合
engine_multi = OCREngine(lang='ch')
```

### 3.3 GPU加速

```python
# 使用GPU加速（需要安装paddlepaddle-gpu）
engine = OCREngine(use_gpu=True)
```

## 4. 命令行使用

### 4.1 单张图片识别

```bash
# 基本识别
python -m ocr_engine recognize image.jpg

# 指定输出文件
python -m ocr_engine recognize image.jpg -o result.json

# 指定语言
python -m ocr_engine recognize image.jpg -l en

# 禁用预处理（提高速度）
python -m ocr_engine recognize image.jpg --no-preprocess

# 使用GPU
python -m ocr_engine recognize image.jpg --gpu
```

### 4.2 批量识别

```bash
# 识别整个文件夹
python -m ocr_engine recognize folder/ -o results.json
```

### 4.3 性能测试

```bash
# 运行性能测试
python -m ocr_engine benchmark --test-data test_images/ -o benchmark.json
```

### 4.4 查看引擎信息

```bash
python -m ocr_engine info
```

## 5. API服务

### 5.1 启动服务

```bash
# 使用uvicorn启动
uvicorn ocr_engine.api.server:app --host 0.0.0.0 --port 8000

# 或使用python模块
python -m ocr_engine.api.server
```

### 5.2 API接口

#### 健康检查

```bash
curl http://localhost:8000/health
```

#### 识别图片

```bash
# 使用文件上传
curl -X POST "http://localhost:8000/recognize" \
  -F "file=@image.jpg" \
  -F "language=ch" \
  -F "enable_preprocessing=true"

# 使用Base64编码
curl -X POST "http://localhost:8000/recognize/base64" \
  -F "image_base64=$(base64 -w 0 image.jpg)" \
  -F "language=ch"
```

#### 仅检测文本区域

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg"
```

#### 识别PDF文件

```bash
# 上传PDF文件进行识别
curl -X POST "http://localhost:8000/recognize/pdf" \
  -F "file=@document.pdf" \
  -F "language=ch" \
  -F "dpi=200"

# 识别指定页面范围
curl -X POST "http://localhost:8000/recognize/pdf" \
  -F "file=@document.pdf" \
  -F "first_page=1" \
  -F "last_page=5"
```

**PDF识别响应示例**：
```json
{
  "success": true,
  "text": "完整的PDF文本内容...",
  "total_pages": 8,
  "processed_pages": 8,
  "page_results": [...],
  "processing_time": 18.5,
  "summary": {
    "total_text_boxes": 170,
    "avg_page_time": 2.31,
    "overall_confidence": 0.972
  }
}
```

### 5.3 API文档

启动服务后，访问以下地址查看交互式API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 6. 结果解析

### 6.1 结果结构

```python
{
    "text": "完整的识别文本",
    "text_boxes": [
        {
            "text": "单个文本框内容",
            "confidence": 0.95,
            "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        }
    ],
    "processing_time": 0.523,
    "image_path": "path/to/image.jpg",
    "metadata": {
        "language": "ch",
        "use_gpu": false,
        "preprocessed": true
    }
}
```

### 6.2 获取置信度统计

```python
result = engine.recognize('image.jpg')
stats = result.get_confidence_stats()
print(f"最小置信度: {stats['min']}")
print(f"最大置信度: {stats['max']}")
print(f"平均置信度: {stats['avg']}")
```

### 6.3 导出结果

```python
import json

result = engine.recognize('image.jpg')

# 导出为JSON
with open('result.json', 'w', encoding='utf-8') as f:
    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

# 导出为纯文本
with open('result.txt', 'w', encoding='utf-8') as f:
    f.write(result.text)
```

## 7. 性能测试

### 7.1 创建测试数据集

```python
from ocr_engine.tests.benchmark import create_test_dataset

# 创建测试数据集
test_dir = create_test_dataset(
    output_dir='test_data',
    num_samples=10,
    texts=['文本1', '文本2', '文本3']
)
```

### 7.2 运行性能测试

```python
from ocr_engine.tests.benchmark import run_benchmark

# 运行测试
results = run_benchmark(
    test_data_path='test_data',
    ground_truth_file='test_data/ground_truth.json',
    lang='ch',
    enable_preprocess=True,
    warmup=3
)

# 保存结果
import json
with open('benchmark_result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### 7.3 测试指标说明

- **CER (Character Error Rate)**: 字符错误率，越低越好
- **WER (Word Error Rate)**: 词错误率，越低越好
- **Throughput**: 吞吐量，每秒处理的图片数
- **Confidence**: 置信度，表示识别结果的可信程度

## 8. 最佳实践

### 8.1 图像质量优化

1. **分辨率**: 建议使用 300 DPI 以上的扫描件
2. **对比度**: 确保文字与背景对比度足够
3. **光照**: 避免过曝或过暗
4. **倾斜**: 尽量保持文档水平

### 8.2 性能优化

1. **预处理选择**:
   - 清晰文档：禁用预处理
   - 模糊文档：启用去噪
   - 光照不均：使用Sauvola二值化

2. **批量处理**:
   ```python
   # 使用批量识别API
   results = engine.recognize_batch(image_list)
   ```

3. **GPU加速**:
   ```python
   engine = OCREngine(use_gpu=True)
   ```

### 8.3 错误处理

```python
from ocr_engine import OCREngine

engine = OCREngine()

try:
    result = engine.recognize('image.jpg')
    if not result.text:
        print("未识别到文字，建议检查图像质量")
    else:
        print(f"识别成功: {result.text}")
except Exception as e:
    print(f"识别失败: {e}")
```

## 9. 常见问题

### Q: 首次运行很慢？
A: 首次运行需要下载模型文件，请耐心等待。模型会缓存到本地，后续运行会更快。

### Q: 中文识别不准确？
A: 请确保：
1. 使用正确的语言设置 `lang='ch'`
2. 图像质量足够清晰
3. 启用预处理功能

### Q: 如何识别手写文字？
A: 当前版本主要针对印刷体优化。手写文字识别建议使用专门的手写识别模型。

### Q: 支持哪些图像格式？
A: 支持 JPG、JPEG、PNG、BMP、TIFF、TIF、WebP 等常见格式。

### Q: 可以识别表格吗？
A: 可以识别表格中的文字，但表格结构解析需要额外处理。

### Q: 支持PDF识别吗？
A: 支持！可以使用以下方式识别PDF：

```python
from ocr_engine import OCREngine

engine = OCREngine()

# 识别完整PDF
result = engine.recognize_pdf('document.pdf')
print(f"总页数: {result.total_pages}")
print(f"文本内容: {result.text}")

# 识别指定页面
result = engine.recognize_pdf(
    'document.pdf',
    first_page=1,
    last_page=5,
    dpi=200
)
```

命令行使用：
```bash
# 识别完整PDF
python -m ocr_engine pdf document.pdf -o result.json

# 识别指定页面
python -m ocr_engine pdf document.pdf --first-page 1 --last-page 5
```

API使用：
```bash
curl -X POST "http://localhost:8000/recognize/pdf" \
  -F "file=@document.pdf" \
  -F "language=ch" \
  -F "dpi=200"
```

## 10. 获取帮助

- 查看API文档: http://localhost:8000/docs
- 查看实现文档: docs/IMPLEMENTATION.md
- 查看技术选型: docs/TECH_SELECTION_REPORT.md
