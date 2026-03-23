"""
FastAPI Web服务
提供HTTP API接口
"""

import io
import base64
from typing import Optional, List
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.engine import OCREngine
from ..core.result import OCRResult, PDFResult


# 数据模型
class OCRRequest(BaseModel):
    """OCR请求模型"""
    language: str = Field(default="ch", description="识别语言")
    enable_preprocessing: bool = Field(default=True, description="是否启用预处理")
    return_details: bool = Field(default=True, description="是否返回详细信息")


class OCRResponse(BaseModel):
    """OCR响应模型"""
    success: bool
    text: str
    text_boxes: List[dict] = []
    processing_time: float
    confidence_stats: dict = {}
    error: Optional[str] = None


class EngineInfoResponse(BaseModel):
    """引擎信息响应模型"""
    version: str
    language: str
    use_gpu: bool
    enable_preprocess: bool
    preprocess_config: dict
    supports_pdf: bool = True


class PDFResponse(BaseModel):
    """PDF识别响应模型"""
    success: bool
    text: str
    total_pages: int
    processed_pages: int
    page_results: List[dict] = []
    processing_time: float
    summary: dict = {}
    error: Optional[str] = None


# 全局引擎实例
_engine: Optional[OCREngine] = None


def get_engine() -> OCREngine:
    """获取或创建OCR引擎实例"""
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="Lightweight OCR API",
        description="基于非大模型技术的高性能OCR服务",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        """根路径"""
        return {
            "name": "Lightweight OCR API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {"status": "healthy"}
    
    @app.get("/info", response_model=EngineInfoResponse)
    async def get_info():
        """获取引擎信息"""
        engine = get_engine()
        info = engine.get_engine_info()
        return EngineInfoResponse(**info)
    
    @app.post("/recognize", response_model=OCRResponse)
    async def recognize(
        file: UploadFile = File(..., description="上传的图片文件"),
        language: str = Form(default="ch", description="识别语言"),
        enable_preprocessing: bool = Form(default=True, description="是否启用预处理")
    ):
        """
        识别上传的图片
        
        - **file**: 图片文件 (JPG, PNG, TIFF, BMP)
        - **language**: 识别语言 (ch, en, ch_en, etc.)
        - **enable_preprocessing**: 是否启用图像预处理
        """
        try:
            # 读取上传的文件
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # 转换为numpy数组
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            
            # 创建引擎并识别
            engine = OCREngine(
                lang=language,
                enable_preprocess=enable_preprocessing
            )
            
            result = engine.recognize(img_array)
            
            return OCRResponse(
                success=True,
                text=result.text,
                text_boxes=[box.to_dict() for box in result.text_boxes],
                processing_time=result.processing_time,
                confidence_stats=result.get_confidence_stats()
            )
            
        except Exception as e:
            return OCRResponse(
                success=False,
                text="",
                processing_time=0.0,
                error=str(e)
            )
    
    @app.post("/recognize/base64", response_model=OCRResponse)
    async def recognize_base64(
        image_base64: str = Form(..., description="Base64编码的图片"),
        language: str = Form(default="ch", description="识别语言"),
        enable_preprocessing: bool = Form(default=True, description="是否启用预处理")
    ):
        """
        识别Base64编码的图片
        
        - **image_base64**: Base64编码的图片字符串
        - **language**: 识别语言
        - **enable_preprocessing**: 是否启用图像预处理
        """
        try:
            # 解码Base64
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # 转换为numpy数组
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            
            # 创建引擎并识别
            engine = OCREngine(
                lang=language,
                enable_preprocess=enable_preprocessing
            )
            
            result = engine.recognize(img_array)
            
            return OCRResponse(
                success=True,
                text=result.text,
                text_boxes=[box.to_dict() for box in result.text_boxes],
                processing_time=result.processing_time,
                confidence_stats=result.get_confidence_stats()
            )
            
        except Exception as e:
            return OCRResponse(
                success=False,
                text="",
                processing_time=0.0,
                error=str(e)
            )
    
    @app.post("/detect")
    async def detect_regions(
        file: UploadFile = File(..., description="上传的图片文件"),
        enable_preprocessing: bool = Form(default=True, description="是否启用预处理")
    ):
        """
        仅检测文本区域（不识别文字）
        
        - **file**: 图片文件
        - **enable_preprocessing**: 是否启用图像预处理
        """
        try:
            # 读取上传的文件
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # 转换为numpy数组
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            
            # 创建引擎并检测
            engine = OCREngine(enable_preprocess=enable_preprocessing)
            regions = engine.detect_text_regions(img_array)
            
            return JSONResponse({
                "success": True,
                "regions": regions,
                "count": len(regions)
            })
            
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": str(e)
            }, status_code=500)
    
    @app.post("/recognize/pdf", response_model=PDFResponse)
    async def recognize_pdf(
        file: UploadFile = File(..., description="上传的PDF文件"),
        language: str = Form(default="ch", description="识别语言"),
        dpi: int = Form(default=200, description="PDF转图像分辨率DPI"),
        first_page: Optional[int] = Form(default=None, description="起始页码(从1开始)"),
        last_page: Optional[int] = Form(default=None, description="结束页码"),
        enable_preprocessing: bool = Form(default=True, description="是否启用预处理")
    ):
        """
        识别上传的PDF文件
        
        - **file**: PDF文件
        - **language**: 识别语言 (ch, en, ch_en, etc.)
        - **dpi**: PDF转图像分辨率，默认200
        - **first_page**: 起始页码，默认从第1页开始
        - **last_page**: 结束页码，默认到最后1页
        - **enable_preprocessing**: 是否启用图像预处理
        """
        try:
            # 验证文件类型
            if not file.filename.lower().endswith('.pdf'):
                return PDFResponse(
                    success=False,
                    text="",
                    total_pages=0,
                    processed_pages=0,
                    processing_time=0.0,
                    error="File must be a PDF"
                )
            
            # 保存上传的PDF文件
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                contents = await file.read()
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            try:
                # 创建引擎并识别PDF
                engine = OCREngine(
                    lang=language,
                    enable_preprocess=enable_preprocessing
                )
                
                result = engine.recognize_pdf(
                    tmp_path,
                    dpi=dpi,
                    first_page=first_page,
                    last_page=last_page
                )
                
                return PDFResponse(
                    success=True,
                    text=result.text,
                    total_pages=result.total_pages,
                    processed_pages=result.processed_pages,
                    page_results=[r.to_dict() for r in result.page_results],
                    processing_time=result.processing_time,
                    summary=result.get_summary()
                )
            finally:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
        except Exception as e:
            return PDFResponse(
                success=False,
                text="",
                total_pages=0,
                processed_pages=0,
                processing_time=0.0,
                error=str(e)
            )
    
    return app


# 用于直接运行的应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
