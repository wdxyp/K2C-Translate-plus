# NLLB Corpus Translator - Colab Version
# 仅用于翻译Corpus Excel文件，适配Google Drive和T4 GPU

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.reader.drawings")

import os
import re
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 配置
NLLB_MODEL_DIR = "/content/drive/MyDrive/translated_models/NLLB200-13B/merged_model"  # 你的merged model路径
NLLB_DEFAULT_BATCH_SIZE = 128
NLLB_DEFAULT_MAX_LENGTH = 96
# 默认Excel文件路径（可选）
DEFAULT_EXCEL_FILE = "/content/drive/MyDrive/Corpus_translate/corpus.xlsx"  # 修改为你的Excel文件路径

# 全局变量
_translation_cache = {}
_nllb_tokenizer = None
_nllb_model = None
_nllb_device = "cuda" if torch.cuda.is_available() else "cpu"
_nllb_loaded_dir = None

# 语言特征正则
HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\ud7b0-\ud7ff]")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def load_nllb_model(model_dir=None):
    """加载NLLB模型"""
    global _nllb_tokenizer, _nllb_model, _nllb_loaded_dir, _nllb_device
    
    if model_dir is None:
        model_dir = NLLB_MODEL_DIR
    
    if _nllb_model is not None and _nllb_loaded_dir == model_dir:
        return _nllb_tokenizer, _nllb_model
    
    print(f"[系统] 正在加载 NLLB 模型: {model_dir}")
    print(f"[系统] 使用设备: {_nllb_device}")
    
    if _nllb_device == "cuda":
        print(f"[系统] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[系统] GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
    
    model = model.to(_nllb_device)
    if _nllb_device == "cuda":
        try:
            model = model.half()
            print(f"[系统] 已启用 FP16 半精度加速")
        except Exception as e:
            print(f"[系统] FP16 加速未启用: {e}")
    model.eval()
    
    _nllb_tokenizer = tokenizer
    _nllb_model = model
    _nllb_loaded_dir = model_dir
    
    print(f"[系统] 模型加载完成!")
    return tokenizer, model


def translate_batch(text_list, src_lang="kor_Hang", tgt_lang="zho_Hans"):
    """批量翻译文本"""
    if not text_list:
        return []
    
    tokenizer, model = load_nllb_model()
    
    batch_size = NLLB_DEFAULT_BATCH_SIZE
    max_length = NLLB_DEFAULT_MAX_LENGTH
    
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict) and tgt_lang in lang_code_to_id:
        forced_bos_token_id = int(lang_code_to_id[tgt_lang])
    else:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    outs = []
    total_batches = (len(text_list) + batch_size - 1) // batch_size
    
    for start in tqdm(range(0, len(text_list), batch_size), desc="翻译进度", total=total_batches):
        batch = text_list[start : start + batch_size]
        
        tokenizer.src_lang = src_lang
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(512, int(getattr(tokenizer, "model_max_length", 512) or 512)),
        )
        encoded = {k: v.to(_nllb_device) for k, v in encoded.items()}
        
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=int(forced_bos_token_id),
                max_length=max_length,
                use_cache=True,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outs.extend([("" if x is None else str(x).strip()) for x in decoded])
    
    return outs


def translate_corpus_file(input_path, output_path=None):
    """翻译Corpus Excel文件：韩文→中文"""
    print(f"\n[系统] 开始处理: {input_path}")
    
    # 读取Excel
    df = pd.read_excel(input_path)
    print(f"[系统] 成功读取，共 {len(df)} 行数据")
    
    # 查找列
    ko_col = None
    zh_col = None
    for col in df.columns:
        col_str = str(col).strip().lower()
        if "ko" in col_str or "韩文" in col_str or "韩" in col_str:
            ko_col = col
        if "zh" in col_str or "中文" in col_str or "中" in col_str:
            zh_col = col
    
    if not ko_col:
        raise ValueError("未找到韩文列，请确保有包含'Ko'、'韩文'或'韩'的列名")
    
    print(f"[系统] 韩文列: {ko_col}")
    if zh_col:
        print(f"[系统] 中文列: {zh_col}")
    else:
        zh_col = "ZH"
        df[zh_col] = ""
        print(f"[系统] 创建新的中文列: {zh_col}")
    
    # 收集需要翻译的文本
    texts_to_translate = []
    indices_to_translate = []
    
    for idx, row in df.iterrows():
        text = str(row[ko_col]) if pd.notna(row[ko_col]) else ""
        if text.strip():
            # 检查缓存
            cache_key = text
            cached = _translation_cache.get(cache_key)
            if cached is not None:
                df.at[idx, zh_col] = cached
            else:
                texts_to_translate.append(text)
                indices_to_translate.append(idx)
    
    print(f"[系统] 待翻译: {len(texts_to_translate)} 句 (已缓存: {len(df) - len(texts_to_translate)} 句)")
    
    if texts_to_translate:
        # 翻译
        translated_texts = translate_batch(texts_to_translate)
        
        # 写回DataFrame
        for idx, trans_text in zip(indices_to_translate, translated_texts):
            df.at[idx, zh_col] = trans_text
            _translation_cache[texts_to_translate[indices_to_translate.index(idx)]] = trans_text
    
    # 保存
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_translated{ext}"
    
    df.to_excel(output_path, index=False)
    print(f"[完成] 文件已保存: {output_path}")
    
    return output_path


# ==========================================
# Colab 便捷使用函数
# ==========================================

def quick_translate(file_path=None):
    """快速翻译函数，一键使用
    
    Args:
        file_path: Excel文件路径（可选，不传则使用默认路径）
    """
    if file_path is None:
        file_path = DEFAULT_EXCEL_FILE
    
    # 先加载模型（显示进度）
    load_nllb_model()
    
    # 翻译
    return translate_corpus_file(file_path)


# ==========================================
# 主程序 - 直接运行即可翻译
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("NLLB Corpus 翻译工具 - Colab 版本")
    print("="*60)
    
    # 使用默认Excel文件，或在这里修改
    input_file = DEFAULT_EXCEL_FILE
    
    print(f"\n准备翻译文件: {input_file}")
    
    # 检查文件是否存在
    if os.path.exists(input_file):
        # 开始翻译
        output_file = quick_translate(input_file)
        print(f"\n✅ 翻译完成！")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
    else:
        print(f"\n❌ 文件不存在: {input_file}")
        print("\n请修改 DEFAULT_EXCEL_FILE 为你的Excel文件路径，")
        print("或使用: quick_translate('/path/to/your/file.xlsx')")
