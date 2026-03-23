"""
PDF识别测试脚本
"""

from ocr_engine import OCREngine
from pathlib import Path
import json

pdf_path = 'test_pdf/1.pdf'

if not Path(pdf_path).exists():
    print(f'PDF文件不存在: {pdf_path}')
else:
    print(f'开始识别PDF: {pdf_path}')
    
    # 创建OCR引擎
    engine = OCREngine(lang='ch')
    
    # 识别PDF
    result = engine.recognize_pdf(pdf_path, dpi=150)
    
    print(f'\n识别完成!')
    print(f'总页数: {result.total_pages}')
    print(f'处理页数: {result.processed_pages}')
    print(f'总处理时间: {result.processing_time:.2f}秒')
    
    # 获取摘要
    summary = result.get_summary()
    print(f'\n摘要信息:')
    print(f"  总文本框数: {summary['total_text_boxes']}")
    print(f"  平均每页时间: {summary['avg_page_time']:.2f}秒")
    print(f"  整体置信度: {summary['overall_confidence']:.3f}")
    
    # 输出每页文本预览
    print(f'\n识别文本预览:')
    for page_result in result.page_results:
        page_num = page_result.metadata.get('page_number', 0)
        print(f'\n--- 第 {page_num} 页 ---')
        text_preview = page_result.text[:300] if len(page_result.text) > 300 else page_result.text
        print(text_preview)
        confidence = page_result.get_confidence_stats()['avg']
        print(f'(置信度: {confidence:.3f})')
    
    # 保存结果到results文件夹
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / 'pdf_result.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.to_json())
    print(f'\n完整结果已保存到: {output_path}')
