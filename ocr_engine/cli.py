"""
命令行接口
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .core.engine import OCREngine
from .utils.image_utils import batch_load_images


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="ocr-cli",
        description="Lightweight OCR Engine - 轻量级OCR命令行工具"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 识别命令
    recognize_parser = subparsers.add_parser("recognize", help="识别图片中的文字")
    recognize_parser.add_argument("input", help="输入图片路径或文件夹")
    recognize_parser.add_argument("-o", "--output", help="输出文件路径")
    recognize_parser.add_argument("-l", "--lang", default="ch", help="识别语言 (默认: ch)")
    recognize_parser.add_argument("--no-preprocess", action="store_true", help="禁用预处理")
    recognize_parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    
    # 测试命令
    benchmark_parser = subparsers.add_parser("benchmark", help="性能测试")
    benchmark_parser.add_argument("--test-data", required=True, help="测试数据文件夹")
    benchmark_parser.add_argument("-o", "--output", help="测试结果输出路径")
    benchmark_parser.add_argument("-l", "--lang", default="ch", help="识别语言")
    
    # PDF识别命令
    pdf_parser = subparsers.add_parser("pdf", help="识别PDF文件")
    pdf_parser.add_argument("input", help="输入PDF文件路径")
    pdf_parser.add_argument("-o", "--output", help="输出文件路径")
    pdf_parser.add_argument("-l", "--lang", default="ch", help="识别语言 (默认: ch)")
    pdf_parser.add_argument("--dpi", type=int, default=200, help="PDF转图像分辨率DPI (默认: 200)")
    pdf_parser.add_argument("--first-page", type=int, help="起始页码")
    pdf_parser.add_argument("--last-page", type=int, help="结束页码")
    pdf_parser.add_argument("--no-preprocess", action="store_true", help="禁用预处理")
    
    # 信息命令
    info_parser = subparsers.add_parser("info", help="显示引擎信息")
    
    return parser


def recognize_command(args) -> int:
    """执行识别命令"""
    input_path = Path(args.input)
    
    # 初始化引擎
    engine = OCREngine(
        use_gpu=args.gpu,
        lang=args.lang,
        enable_preprocess=not args.no_preprocess
    )
    
    results = []
    
    if input_path.is_file():
        # 单文件识别
        print(f"识别图片: {input_path}")
        result = engine.recognize(str(input_path))
        results.append({
            "file": str(input_path),
            "result": result.to_dict()
        })
        print(f"\n识别结果:\n{result.text}")
        print(f"处理时间: {result.processing_time:.3f}s")
        
    elif input_path.is_dir():
        # 批量识别
        print(f"批量识别文件夹: {input_path}")
        images = batch_load_images(input_path)
        
        for i, (img_path, _) in enumerate(images):
            print(f"处理 {i+1}/{len(images)}: {img_path}")
            result = engine.recognize(img_path)
            results.append({
                "file": img_path,
                "result": result.to_dict()
            })
    else:
        print(f"错误: 路径不存在 - {input_path}")
        return 1
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_path}")
    
    return 0


def benchmark_command(args) -> int:
    """执行性能测试命令"""
    from .tests.benchmark import run_benchmark
    
    print(f"开始性能测试，测试数据: {args.test_data}")
    results = run_benchmark(args.test_data, lang=args.lang)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"测试结果已保存到: {output_path}")
    
    return 0


def pdf_command(args) -> int:
    """执行PDF识别命令"""
    from .utils.image_utils import is_pdf_file
    
    input_path = Path(args.input)
    
    # 检查文件是否存在
    if not input_path.exists():
        print(f"错误: 文件不存在 - {input_path}")
        return 1
    
    # 检查是否为PDF文件
    if not is_pdf_file(input_path):
        print(f"错误: 不是PDF文件 - {input_path}")
        return 1
    
    # 初始化引擎
    engine = OCREngine(
        lang=args.lang,
        enable_preprocess=not args.no_preprocess
    )
    
    print(f"识别PDF文件: {input_path}")
    print(f"参数: dpi={args.dpi}, language={args.lang}")
    if args.first_page:
        print(f"      first_page={args.first_page}")
    if args.last_page:
        print(f"      last_page={args.last_page}")
    
    # 识别PDF
    result = engine.recognize_pdf(
        str(input_path),
        dpi=args.dpi,
        first_page=args.first_page,
        last_page=args.last_page
    )
    
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
    
    # 输出文本预览
    print(f"\n识别文本预览:")
    for page_result in result.page_results[:3]:  # 只显示前3页
        page_num = page_result.metadata.get('page_number', 0)
        print(f"\n--- 第 {page_num} 页 ---")
        text_preview = page_result.text[:200] if len(page_result.text) > 200 else page_result.text
        print(text_preview)
    
    if len(result.page_results) > 3:
        print(f"\n... 还有 {len(result.page_results) - 3} 页 ...")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        print(f"\n完整结果已保存到: {output_path}")
    
    return 0


def info_command(args) -> int:
    """显示引擎信息"""
    engine = OCREngine()
    info = engine.get_engine_info()
    
    print("OCR Engine Information")
    print("=" * 40)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    return 0


def main() -> int:
    """主入口函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        if args.command == "recognize":
            return recognize_command(args)
        elif args.command == "benchmark":
            return benchmark_command(args)
        elif args.command == "pdf":
            return pdf_command(args)
        elif args.command == "info":
            return info_command(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
