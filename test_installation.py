#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
依赖包安装测试脚本

用法：
    python test_installation.py
    python test_installation.py --stage 3  # 只测试第三阶段
    python test_installation.py --all      # 测试所有包
"""

import sys
import argparse


def test_package(package_name, display_name=None):
    """测试单个包是否已安装"""
    if display_name is None:
        display_name = package_name

    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {display_name:30s} {version}")
        return True
    except ImportError:
        print(f"❌ {display_name:30s} 未安装")
        return False


def test_stage_basic():
    """测试基础包（所有阶段通用）"""
    print("\n" + "="*60)
    print("基础包（通用）")
    print("="*60)

    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
    ]

    results = []
    for pkg, name in packages:
        results.append(test_package(pkg, name))

    return all(results)


def test_stage_1():
    """测试第一阶段：Python 基础"""
    print("\n" + "="*60)
    print("第一阶段：Python 基础")
    print("="*60)

    packages = [
        ('starlette', 'Starlette'),
        ('uvicorn', 'Uvicorn'),
    ]

    results = []
    for pkg, name in packages:
        results.append(test_package(pkg, name))

    return all(results)


def test_stage_3():
    """测试第三阶段：数据分析"""
    print("\n" + "="*60)
    print("第三阶段：数据分析")
    print("="*60)

    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('plotly', 'Plotly'),
        ('pymysql', 'PyMySQL'),
        ('sqlalchemy', 'SQLAlchemy'),
    ]

    results = []
    for pkg, name in packages:
        results.append(test_package(pkg, name))

    return all(results)


def test_stage_4():
    """测试第四阶段：机器学习"""
    print("\n" + "="*60)
    print("第四阶段：机器学习")
    print("="*60)

    packages = [
        ('sklearn', 'Scikit-learn'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('shap', 'SHAP'),
    ]

    results = []
    for pkg, name in packages:
        results.append(test_package(pkg, name))

    return all(results)


def test_stage_5():
    """测试第五阶段：深度学习"""
    print("\n" + "="*60)
    print("第五阶段：深度学习")
    print("="*60)

    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('torchaudio', 'TorchAudio'),
        ('tensorboard', 'TensorBoard'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
    ]

    results = []
    for pkg, name in packages:
        result = test_package(pkg, name)
        results.append(result)

        # 检查 CUDA 支持
        if pkg == 'torch' and result:
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                cuda_version = torch.version.cuda if cuda_available else 'N/A'
                print(f"   CUDA 支持: {cuda_available} (CUDA {cuda_version})")
            except:
                pass

    return all(results)


def test_stage_6():
    """测试第六阶段：NLP"""
    print("\n" + "="*60)
    print("第六阶段：NLP")
    print("="*60)

    packages = [
        ('transformers', 'Transformers'),
        ('tokenizers', 'Tokenizers'),
        ('datasets', 'Datasets'),
        ('nltk', 'NLTK'),
        ('spacy', 'SpaCy'),
        ('jieba', 'Jieba'),
        ('sentencepiece', 'SentencePiece'),
        ('fasttext', 'FastText'),
    ]

    results = []
    for pkg, name in packages:
        results.append(test_package(pkg, name))

    return all(results)


def test_stage_7():
    """测试第七阶段：大模型 & 多模态"""
    print("\n" + "="*60)
    print("第七阶段：大模型 & 多模态")
    print("="*60)

    packages = [
        ('peft', 'PEFT'),
        ('accelerate', 'Accelerate'),
        ('timm', 'Timm'),
        ('chromadb', 'ChromaDB'),
        ('faiss', 'Faiss'),
        ('langchain', 'LangChain'),
        ('optimum', 'Optimum'),
    ]

    results = []
    for pkg, name in packages:
        results.append(test_package(pkg, name))

    # 可选包
    print("\n  可选包（需要特定环境）：")
    optional_packages = [
        ('bitsandbytes', 'BitsAndBytes'),
        ('deepspeed', 'DeepSpeed'),
    ]

    for pkg, name in optional_packages:
        test_package(pkg, name)

    return all(results)


def test_stage_8():
    """测试第八阶段：强化学习"""
    print("\n" + "="*60)
    print("第八阶段：强化学习")
    print("="*60)

    packages = [
        ('gym', 'Gym'),
        ('stable_baselines3', 'Stable-Baselines3'),
    ]

    results = []
    for pkg, name in packages:
        results.append(test_package(pkg, name))

    return all(results)


def test_stage_9():
    """测试第九阶段：大模型应用"""
    print("\n" + "="*60)
    print("第九阶段：大模型应用")
    print("="*60)

    packages = [
        ('fastapi', 'FastAPI'),
        ('pydantic', 'Pydantic'),
        ('gradio', 'Gradio'),
        ('streamlit', 'Streamlit'),
    ]

    results = []
    for pkg, name in packages:
        results.append(test_package(pkg, name))

    return all(results)


def test_tools():
    """测试通用工具"""
    print("\n" + "="*60)
    print("通用工具")
    print("="*60)

    packages = [
        ('jupyter', 'Jupyter'),
        ('tqdm', 'Tqdm'),
        ('dotenv', 'Python-Dotenv'),
        ('yaml', 'PyYAML'),
        ('pytest', 'Pytest'),
    ]

    results = []
    for pkg, name in packages:
        results.append(test_package(pkg, name))

    return all(results)


def main():
    parser = argparse.ArgumentParser(description='测试依赖包安装')
    parser.add_argument('--stage', type=int, help='测试指定阶段 (1-9)')
    parser.add_argument('--all', action='store_true', help='测试所有包')
    args = parser.parse_args()

    print("="*60)
    print("依赖包安装测试")
    print("="*60)
    print(f"Python 版本: {sys.version}")
    print("="*60)

    if args.stage:
        # 测试指定阶段
        stage_functions = {
            1: test_stage_1,
            3: test_stage_3,
            4: test_stage_4,
            5: test_stage_5,
            6: test_stage_6,
            7: test_stage_7,
            8: test_stage_8,
            9: test_stage_9,
        }

        if args.stage in stage_functions:
            stage_functions[args.stage]()
        else:
            print(f"❌ 无效的阶段: {args.stage}")
            print("   可用阶段: 1, 3, 4, 5, 6, 7, 8, 9")

    elif args.all:
        # 测试所有包
        test_stage_basic()
        test_stage_1()
        test_stage_3()
        test_stage_4()
        test_stage_5()
        test_stage_6()
        test_stage_7()
        test_stage_8()
        test_stage_9()
        test_tools()

    else:
        # 默认只测试基础包
        test_stage_basic()

        print("\n" + "="*60)
        print("提示：")
        print("  - 运行 'python test_installation.py --stage 3' 测试指定阶段")
        print("  - 运行 'python test_installation.py --all' 测试所有包")
        print("="*60)

    print("\n测试完成！\n")


if __name__ == '__main__':
    main()
