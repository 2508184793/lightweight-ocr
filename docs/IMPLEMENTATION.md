# OCR项目实现文档

## 1. 项目架构

### 1.1 整体架构

```
ocr_engine/
├── __init__.py              # 包入口
├── cli.py                   # 命令行接口
├── core/                    # 核心模块
│   ├── __init__.py
│   ├── engine.py           # OCR引擎主类
│   └── result.py           # 结果数据模型
├── preprocessing/           # 预处理模块
│   ├── __init__.py
│   └── preprocessor.py     # 图像预处理器
├── utils/                   # 工具模块
│   ├── __init__.py
│   └── image_utils.py      # 图像工具函数
├── api/                     # API服务模块
│   ├── __init__.py
│   └── server.py           # FastAPI服务
└── tests/                   # 测试模块
    ├── __init__.py
    └── benchmark.py        # 性能测试
```

### 1.2 核心组件说明

#### OCREngine（核心引擎）
- **文件**: `core/engine.py`
- **功能**: 封装PaddleOCR，提供统一的识别接口
- **主要方法**:
  - `recognize()`: 单张图片识别
  - `recognize_batch()`: 批量识别
  - `detect_text_regions()`: 仅检测文本区域

#### ImagePreprocessor（图像预处理器）
- **文件**: `preprocessing/preprocessor.py`
- **功能**: 提供图像预处理功能
- **处理流程**:
  1. 灰度转换
  2. 对比度增强（CLAHE）
  3. 去噪（高斯/中值/双边滤波）
  4. 二值化（Otsu/自适应/Sauvola）
  5. 倾斜校正（霍夫变换）

#### OCRResult（结果模型）
- **文件**: `core/result.py`
- **功能**: 定义OCR结果数据结构
- **包含信息**:
  - 识别文本
  - 文本框位置
  - 置信度
  - 处理时间
  - 元数据

## 2. 技术实现细节

### 2.1 图像预处理算法

#### 去噪算法

**高斯滤波**
```python
# 使用高斯核平滑图像，去除高斯噪声
blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```

**中值滤波**
```python
# 使用邻域中值替换像素，去除椒盐噪声
blurred = cv2.medianBlur(image, kernel_size)
```

**双边滤波**
```python
# 保持边缘的同时去噪
blurred = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
```

#### 二值化算法

**Otsu自动阈值**
```python
# 自动计算全局阈值
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

**自适应阈值**
```python
# 局部自适应阈值
binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, block_size, C)
```

**Sauvola算法**
```python
# 适合光照不均的文档
threshold = mean * (1 + k * (std / R - 1))
```

#### 倾斜校正

使用霍夫变换检测文本行角度：
```python
# 边缘检测
edges = cv2.Canny(image, 50, 150)

# 霍夫变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# 计算中位数角度并旋转
median_angle = np.median(angles)
M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
```

### 2.2 OCR引擎集成

#### PaddleOCR配置

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,    # 使用方向分类器
    lang='ch',             # 中文模型
    use_gpu=False,         # CPU推理
    show_log=False         # 关闭冗余日志
)
```

#### 识别流程

1. 加载图像
2. 图像预处理（可选）
3. 文本检测（DBNet算法）
4. 方向分类
5. 文本识别（SVTR_LCNet）
6. 结果解析与格式化

### 2.3 性能评估指标

#### 字符错误率（CER）

使用Levenshtein距离计算：
```
CER = (插入数 + 删除数 + 替换数) / 总字符数
```

#### 词错误率（WER）

```
WER = (插入词 + 删除词 + 替换词) / 总词数
```

#### 处理速度

- 单张图片处理时间
- 吞吐量（张/秒）
- 平均置信度

## 3. API设计

### 3.1 Python API

```python
from ocr_engine import OCREngine

# 创建引擎
engine = OCREngine(
    use_gpu=False,
    lang='ch',
    enable_preprocess=True
)

# 识别图片
result = engine.recognize('image.jpg')
print(result.text)
print(result.processing_time)
```

### 3.2 REST API

#### 识别接口

```http
POST /recognize
Content-Type: multipart/form-data

file: <图片文件>
language: ch
enable_preprocessing: true
```

**响应**:
```json
{
  "success": true,
  "text": "识别结果",
  "text_boxes": [...],
  "processing_time": 0.523,
  "confidence_stats": {
    "min": 0.89,
    "max": 0.98,
    "avg": 0.93
  }
}
```

#### 检测接口

```http
POST /detect
Content-Type: multipart/form-data

file: <图片文件>
```

**响应**:
```json
{
  "success": true,
  "regions": [...],
  "count": 5
}
```

### 3.3 命令行接口

```bash
# 单张识别
ocr-cli recognize image.jpg -o result.json

# 批量识别
ocr-cli recognize folder/ -o results.json

# 性能测试
ocr-cli benchmark --test-data test_images/ -o benchmark.json

# 查看信息
ocr-cli info
```

## 4. 配置说明

### 4.1 预处理配置

```python
preprocess_config = {
    "denoise_method": "gaussian",      # 去噪方法
    "binarize_method": "otsu",         # 二值化方法
    "enable_deskew": True,             # 启用倾斜校正
    "enable_contrast_enhance": True,   # 启用对比度增强
    "denoise_kernel_size": 5,          # 去噪核大小
    "binarize_block_size": 11,         # 二值化块大小
    "contrast_clip_limit": 2.0,        # 对比度限制
    "contrast_grid_size": (8, 8)       # 对比度网格大小
}
```

### 4.2 引擎配置

```python
engine_config = {
    "use_gpu": False,          # 是否使用GPU
    "lang": "ch",              # 识别语言
    "enable_preprocess": True, # 是否启用预处理
    "preprocess_config": {...} # 预处理配置
}
```

## 5. 扩展开发

### 5.1 添加新的预处理方法

在 `preprocessing/preprocessor.py` 中添加：

```python
def _custom_preprocess(self, image: np.ndarray) -> np.ndarray:
    # 实现自定义预处理逻辑
    return processed_image
```

### 5.2 添加新的输出格式

在 `core/result.py` 的 `OCRResult` 类中添加：

```python
def to_custom_format(self) -> str:
    # 转换为自定义格式
    return formatted_string
```

### 5.3 集成其他OCR引擎

创建新的引擎类：

```python
class CustomOCREngine:
    def recognize(self, image) -> OCRResult:
        # 实现识别逻辑
        return OCRResult(...)
```

## 6. 性能优化建议

### 6.1 图像预处理优化

- 对于清晰文档，可禁用预处理以提高速度
- 根据图像质量选择合适的去噪方法
- 对于大批量处理，考虑使用批量预处理

### 6.2 推理优化

- 使用GPU加速（如果可用）
- 调整PaddleOCR的推理参数
- 使用模型量化减少内存占用

### 6.3 系统优化

- 使用多进程处理大批量图片
- 实现结果缓存机制
- 使用异步API提高并发能力

## 7. 故障排除

### 7.1 常见问题

**问题**: 识别结果为空
- **解决**: 检查图像质量，启用预处理，调整二值化方法

**问题**: 处理速度慢
- **解决**: 禁用预处理，使用GPU，减小图像尺寸

**问题**: 中文识别准确率低
- **解决**: 确保使用正确的语言模型（lang='ch'）

### 7.2 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 8. 版本历史

### v1.0.0 (2024-03)
- 初始版本发布
- 集成PaddleOCR PP-OCRv4
- 实现完整的预处理流程
- 提供Python API、REST API和CLI
- 包含性能测试模块
