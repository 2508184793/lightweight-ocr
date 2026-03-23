"""
PDF识别示例
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_engine import OCREngine


def example_pdf_recognition():
    """PDF识别示例"""
    print("=" * 50)
    print("PDF识别示例")
    print("=" * 50)
    
    # 创建OCR引擎
    engine = OCREngine(lang='ch')
    
    # 示例：请替换为您的PDF文件路径
    pdf_path = "path/to/your/document.pdf"
    
    if not Path(pdf_path).exists():
        print(f"示例PDF文件不存在: {pdf_path}")
        print("请替换为实际的PDF文件路径后运行此示例")
        return
    
    # 识别PDF（识别所有页面）
    print(f"开始识别PDF: {pdf_path}")
    result = engine.recognize_pdf(pdf_path, dpi=200)
    
    # 输出结果
    print(f"\n识别完成!")
    print(f"总页数: {result.total_pages}")
    print(f"处理页数: {result.processed_pages}")
    print(f"总处理时间: {result.processing_time:.2f}秒")
    
    # 获取摘要
    summary = result.get_summary()
    print(f"\n摘要信息:")
    print(f"  总文本框数: {summary['total_text_boxes']}")
    print(f"  平均每页时间: {summary['avg_page_time']:.2f}秒")
    print(f"  整体置信度: {summary['overall_confidence']:.3f}")
    
    # 输出前几页文本
    print(f"\n识别文本预览（前3页）:")
    for page_result in result.page_results[:3]:
        page_num = page_result.metadata.get('page_number', 0)
        print(f"\n--- 第 {page_num} 页 ---")
        print(page_result.text[:200] + "..." if len(page_result.text) > 200 else page_result.text)
    
    # 保存结果
    output_path = "pdf_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.to_json())
    print(f"\n完整结果已保存到: {output_path}")


def example_pdf_specific_pages():
    """识别PDF指定页面示例"""
    print("\n" + "=" * 50)
    print("PDF指定页面识别示例")
    print("=" * 50)
    
    engine = OCREngine(lang='ch')
    pdf_path = "path/to/your/document.pdf"
    
    if not Path(pdf_path).exists():
        print(f"示例PDF文件不存在: {pdf_path}")
        return
    
    # 只识别第1-3页
    print(f"识别PDF的第1-3页: {pdf_path}")
    result = engine.recognize_pdf(
        pdf_path,
        dpi=200,
        first_page=1,
        last_page=3
    )
    
    print(f"\n识别完成!")
    print(f"PDF总页数: {result.total_pages}")
    print(f"本次处理页数: {result.processed_pages}")
    
    # 获取特定页面的文本
    page_2_text = result.get_page_text(2)
    print(f"\n第2页文本:\n{page_2_text[:300]}...")


def example_pdf_info():
    """获取PDF信息示例"""
    print("\n" + "=" * 50)
    print("PDF信息获取示例")
    print("=" * 50)
    
    from ocr_engine.utils.image_utils import get_pdf_info
    
    pdf_path = "path/to/your/document.pdf"
    
    if not Path(pdf_path).exists():
        print(f"示例PDF文件不存在: {pdf_path}")
        return
    
    info = get_pdf_info(pdf_path)
    print(f"PDF信息:")
    print(f"  路径: {info['path']}")
    print(f"  页数: {info['page_count']}")
    print(f"  文件大小: {info['file_size'] / 1024:.2f} KB")
    print(f"  元数据: {info['metadata']}")


if __name__ == "__main__":
    print("PDF识别功能示例")
    print("=" * 50)
    print("\n注意：请将示例中的 'path/to/your/document.pdf' 替换为实际的PDF文件路径")
    print("\n可用的示例函数:")
    print("  1. example_pdf_recognition() - 识别完整PDF")
    print("  2. example_pdf_specific_pages() - 识别指定页面")
    print("  3. example_pdf_info() - 获取PDF信息")
    print("\n使用示例:")
    print("  修改 pdf_path 变量后运行相应函数")
