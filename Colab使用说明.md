# Colab NLLB 翻译工具使用说明

## 快速开始

### 1. 准备工作

在 Google Drive 上创建以下结构：

```
My Drive/
└── translated_models/
    └── NLLB200-13B/
        └── merged_model/
            ├── config.json
            ├── pytorch_model.bin
            └── ...
```

### 2. Colab 代码模板

在 Colab 中创建一个新笔记本，按以下步骤执行：

#### 第1步：挂载 Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 第2步：添加代码路径
将 `Colab_NLLB_Translator.py` 放到 Google Drive 的 `Files_share(Python)` 文件夹中，然后：

```python
import sys
sys.path.append('/content/drive/MyDrive/Files_share(Python)')

# 或者直接复制代码到单元格
```

#### 第3步：安装依赖
```python
!pip install transformers pandas openpyxl tqdm sentencepiece
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 第4步：检查 GPU
```python
import torch
print(f"GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
```

#### 第5步：开始翻译

**方式A - 使用便捷函数（推荐）：**
```python
from Colab_NLLB_Translator import quick_translate

# 翻译
output_file = quick_translate('/content/drive/MyDrive/corpus.xlsx')
```

**方式B - 更细粒度控制：**
```python
from Colab_NLLB_Translator import load_nllb_model, translate_corpus_file

# 先加载模型
load_nllb_model()
# 或指定路径
load_nllb_model('/content/drive/MyDrive/your-custom-model-path')

# 翻译
translate_corpus_file(
    '/content/drive/MyDrive/input.xlsx',
    '/content/drive/MyDrive/output_translated.xlsx'
)
```

### 3. 文件格式要求

Excel文件需要包含：
- **韩文列**：列名包含 "Ko"、"韩文" 或 "韩"
- **中文列（可选）**：列名包含 "Zh"、"中文" 或 "中"。如果没有会自动创建"ZH"列

### 4. 配置说明

可以修改 `Colab_NLLB_Translator.py` 中的配置：

```python
NLLB_MODEL_DIR = "/content/drive/MyDrive/translated_models/NLLB200-13B/merged_model"  # 模型路径
NLLB_DEFAULT_BATCH_SIZE = 64  # 批处理大小，显存够大可以调到128
NLLB_DEFAULT_MAX_LENGTH = 96  # 最大生成长度
```

### 5. 性能提示

- **T4 GPU**: 1000句预计 2-5分钟
- **Batch Size**: 如果显存允许，建议增大到 128
- **FP16**: 已自动启用半精度加速

### 6. 常见问题

**Q: 找不到模型文件？**
A: 检查模型路径是否正确，确保是包含 config.json 的目录

**Q: 显存不足？**
A: 减小 `NLLB_DEFAULT_BATCH_SIZE` 到 32 或 16

**Q: 翻译速度慢？**
A: 确保已启用GPU，看输出中是否有 `[系统] 检测到 GPU:`

## 完整示例 Notebooks

### 完整一键运行代码

```python
# ============================================
# 完整 Colab 一键运行脚本
# ============================================

# 1. 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 安装依赖
!pip install transformers pandas openpyxl tqdm sentencepiece
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 导入翻译模块（文件在 /content/drive/MyDrive/Files_share(Python)）
import sys
sys.path.append('/content/drive/MyDrive/Files_share(Python)')

# 4. 检查GPU
import torch
print(f"GPU可用: {torch.cuda.is_available()}")

# 5. 开始翻译
from Colab_NLLB_Translator import quick_translate

# 设置你的文件路径
input_file = '/content/drive/MyDrive/your_corpus.xlsx'

# 开始翻译！
output_file = quick_translate(input_file)

print(f"\n翻译完成！输出文件: {output_file}")
```
