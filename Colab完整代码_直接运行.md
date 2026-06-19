# 🚀 Colab 完整运行代码 - 一键复制即可

## 完整 Colab 笔记本代码 - 全部复制到 Colab 单元格里运行！

---

## 📋 完整代码：

```python
# ============================================
# 🚀 第1步：挂载 Google Drive
# ============================================
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive 挂载完成！")

# ============================================
# 📦 第2步：安装依赖
# ============================================
print("\n正在安装依赖...")
!pip install -q transformers pandas openpyxl tqdm sentencepiece
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
print("✅ 依赖安装完成！")

# ============================================
# 📂 第3步：添加代码路径
# ============================================
import sys
sys.path.append('/content/drive/MyDrive/Files_share(Python)')
print("✅ 代码路径添加完成！")

# ============================================
# 🖥️ 第4步：检查 GPU
# ============================================
import torch
print(f"\nGPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  未检测到 GPU，将使用 CPU（速度较慢）")

# ============================================
# 🚀 第5步：开始翻译
# ============================================
from Colab_NLLB_Translator import quick_translate

# 设置你的 Excel 文件路径
# 修改下面的路径为你实际的文件路径
input_file = "/content/drive/MyDrive/corpus.xlsx"

print(f"\n准备翻译文件: {input_file}")
print("="*60)

# 开始翻译！
output_file = quick_translate(input_file)

print("="*60)
print(f"\n✅ 翻译完成！")
print(f"输入文件: {input_file}")
print(f"输出文件: {output_file}")
```

---

## 📂 预期的文件结构（请确保你的 Drive 中有这些文件：

```
My Drive/
├── Files_share(Python)/          # ← 放这里
│   └── Colab_NLLB_Translator.py
├── translated_models/
│   └── NLLB200-13B/
│       └── merged_model/
│           ├── config.json
│           ├── pytorch_model.bin
│           └── ...
└── corpus.xlsx                  # ← 你的 Excel 文件
```

---

## 💡 使用说明：

1. **把上面的完整代码复制到 Colab 的一个单元格里
2. **修改 `input_file` 变量为你实际的 Excel 文件路径
3. **点击运行按钮即可！

---

## ⚙️ 如果需要修改配置：

如果你想修改默认配置，编辑 `Colab_NLLB_Translator.py` 里的：

```python
NLLB_DEFAULT_BATCH_SIZE = 64  # 批处理大小（显存够大可以改 128）
NLLB_DEFAULT_MAX_LENGTH = 96  # 最大生成长度
```

---

## 📊 性能参考：

- **T4 GPU + Batch 64**: 1000句 ≈ 2-5分钟
- **Batch 翻译进度条会显示实时进度！
