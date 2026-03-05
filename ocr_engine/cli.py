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
