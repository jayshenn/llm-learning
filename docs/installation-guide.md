# 依赖安装指南

本文档提供详细的依赖包安装说明，帮助你按阶段安装所需的 Python 包。

## 目录

- [安装前准备](#安装前准备)
- [推荐安装方式](#推荐安装方式)
- [分阶段安装](#分阶段安装)
- [常见问题](#常见问题)

---

## 安装前准备

### 1. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# macOS/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 2. 升级 pip

```bash
pip install --upgrade pip
```

### 3. 检查 Python 版本

```bash
python --version  # 需要 Python 3.8+
```

---

## 推荐安装方式

### 方式一：分阶段安装（推荐）

根据学习进度，只安装当前阶段需要的包。

### 方式二：安装基础包

```bash
# 安装核心依赖（数据分析 + 机器学习）
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 方式三：完整安装（不推荐）

```bash
# 安装所有依赖（可能遇到编译问题）
pip install -r requirements.txt
```

---

## 分阶段安装

### 🔹 第一阶段：Python 基础

```bash
pip install starlette uvicorn[standard]
```

**验证安装：**

```bash
python -c "import starlette; print(starlette.__version__)"
```

---

### 🔹 第三阶段：数据分析

```bash
# 基础包
pip install numpy pandas scipy

# 可视化
pip install matplotlib seaborn plotly

# 数据库
pip install pymysql sqlalchemy

# Jupyter
pip install jupyter notebook ipywidgets
```

**验证安装：**

```bash
python -c "import pandas as pd; print(pd.__version__)"
jupyter --version
```

---

### 🔹 第四阶段：机器学习

```bash
# 机器学习框架
pip install scikit-learn xgboost lightgbm

# 模型解释
pip install shap
```

**验证安装：**

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

---

### 🔹 第五阶段：深度学习

#### CPU 版本（macOS）

```bash
# 安装 PyTorch CPU 版本
pip install torch torchvision torchaudio

# 深度学习工具
pip install tensorboard torchinfo

# 图像处理
pip install pillow opencv-python
```

#### GPU 版本（CUDA 11.8）

```bash
# 安装 PyTorch GPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 深度学习工具
pip install tensorboard torchinfo

# 图像处理
pip install pillow opencv-python
```

#### GPU 版本（CUDA 12.1）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**验证安装：**

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

### 🔹 第六阶段：NLP

```bash
# Hugging Face 生态
pip install transformers tokenizers datasets

# NLP 工具
pip install nltk spacy jieba

# 文本处理
pip install sentencepiece sacremoses

# fastText
pip install fasttext
```

**首次使用 NLTK 需要下载数据：**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**首次使用 SpaCy 需要下载模型：**

```bash
# 英文模型
python -m spacy download en_core_web_sm

# 中文模型
python -m spacy download zh_core_web_sm
```

**验证安装：**

```bash
python -c "import transformers; print(transformers.__version__)"
```

---

### 🔹 第七阶段：大模型 & 多模态

```bash
# 大模型训练（需要先安装 PyTorch）
pip install peft accelerate

# 多模态
pip install timm

# 向量数据库
pip install chromadb faiss-cpu

# RAG 相关
pip install langchain llama-index

# 模型量化
pip install optimum
```

#### 可选包（需要 CUDA 和编译环境）

**bitsandbytes**（仅 Linux + CUDA）

```bash
pip install bitsandbytes
```

**auto-gptq**（需要 CUDA 和编译工具）

```bash
# 确保已安装 PyTorch
pip install auto-gptq
```

**deepspeed**（需要 CUDA）

```bash
pip install deepspeed
```

**faiss-gpu**（如果有 GPU）

```bash
# 先卸载 CPU 版本
pip uninstall faiss-cpu

# 安装 GPU 版本
pip install faiss-gpu
```

**验证安装：**

```bash
python -c "import peft; print(peft.__version__)"
python -c "import chromadb; print(chromadb.__version__)"
```

---

### 🔹 第八阶段：强化学习

```bash
# 强化学习
pip install gym stable-baselines3

# 或使用新版本
pip install gymnasium
```

**验证安装：**

```bash
python -c "import gym; print(gym.__version__)"
```

---

### 🔹 第九阶段：大模型应用

```bash
# API 开发
pip install fastapi pydantic python-multipart

# 前端界面
pip install gradio streamlit
```

**验证安装：**

```bash
python -c "import gradio; print(gradio.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
```

---

## 常见问题

### 1. auto-gptq 安装失败

**错误：** `Building cuda extension requires PyTorch`

**解决方法：**

```bash
# 方法一：先安装 PyTorch
pip install torch torchvision torchaudio
pip install auto-gptq

# 方法二：跳过 auto-gptq（不是必需的）
# 直接注释掉或不安装
```

---

### 2. bitsandbytes 在 macOS/Windows 上安装失败

**原因：** bitsandbytes 主要支持 Linux + CUDA 环境

**解决方法：**

- macOS: 不需要安装，大部分功能可以用其他方式替代
- Windows: 可以尝试社区版本或跳过

---

### 3. deepspeed 编译失败

**原因：** deepspeed 需要 CUDA 和特定的编译工具链

**解决方法：**

```bash
# 确保安装了 CUDA 开发工具
# Linux:
sudo apt-get install build-essential

# 或者跳过 deepspeed（不是必需的）
```

---

### 4. torch 安装速度慢

**解决方法：** 使用国内镜像源

```bash
# 使用清华镜像
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 5. 检查 CUDA 版本

```bash
# 检查系统 CUDA 版本
nvidia-smi

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"
```

---

## 通用工具安装

这些工具对所有阶段都有用：

```bash
# Jupyter
pip install jupyter notebook ipywidgets

# 进度条
pip install tqdm

# 配置管理
pip install python-dotenv pyyaml

# 日志
pip install loguru

# 测试
pip install pytest pytest-cov

# 代码质量
pip install black flake8 isort
```

---

## 最小安装方案

如果只是想快速开始学习，可以只安装最小依赖：

```bash
# 核心包
pip install numpy pandas matplotlib scikit-learn jupyter

# 根据需要逐步添加
```

---

## 验证完整安装

创建一个测试脚本 `test_installation.py`：

```python
import sys

packages = [
    'numpy',
    'pandas',
    'matplotlib',
    'sklearn',
    'torch',
    'transformers',
]

print("检查已安装的包：\n")
for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package:20s} {version}")
    except ImportError:
        print(f"❌ {package:20s} 未安装")
```

运行测试：

```bash
python test_installation.py
```

---

## 导出当前环境

如果你想保存当前已安装的包：

```bash
# 导出所有包
pip freeze > my_requirements.txt

# 只导出项目需要的包（推荐）
pip list --format=freeze > my_requirements.txt
```

---

## 更新说明

- 本指南会随着项目进展持续更新
- 如果遇到问题，请查看 [discuss/](../discuss/) 目录下的讨论文档
- 建议定期更新依赖包：`pip install --upgrade <package-name>`
