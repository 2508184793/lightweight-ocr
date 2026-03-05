"""
OCR结果数据模型
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class TextBox:
    """文本框信息"""
    text: str
    confidence: float
    box: List[List[int]]  # 四个角点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "box": self.box
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBox":
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            box=data["box"]
        )


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str  # 完整文本（所有文本框合并）
    text_boxes: List[TextBox] = field(default_factory=list)
    processing_time: float = 0.0  # 处理时间（秒）
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.text and self.text_boxes:
            self.text = "\n".join([box.text for box in self.text_boxes])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "text_boxes": [box.to_dict() for box in self.text_boxes],
            "processing_time": self.processing_time,
            "image_path": self.image_path,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRResult":
        text_boxes = [TextBox.from_dict(box) for box in data.get("text_boxes", [])]
        return cls(
            text=data.get("text", ""),
            text_boxes=text_boxes,
            processing_time=data.get("processing_time", 0.0),
            image_path=data.get("image_path"),
            metadata=data.get("metadata", {})
        )
    
    def get_confidence_stats(self) -> Dict[str, float]:
        """获取置信度统计信息"""
        if not self.text_boxes:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        
        confidences = [box.confidence for box in self.text_boxes]
        return {
            "min": min(confidences),
            "max": max(confidences),
            "avg": sum(confidences) / len(confidences)
        }
