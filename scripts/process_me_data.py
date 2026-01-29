import pandas as pd
import glob
import os
import re
from difflib import SequenceMatcher
try:
    import jieba
    import jieba.analyse
except ImportError:
    print("jieba not found. Please install it using: pip install jieba")
    jieba = None

try:
    from openai import OpenAI
except ImportError:
    print("openai not found. Please install it using: pip install openai")
    OpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # 如果没有安装 tqdm，则使用占位符

# ==========================================
# 配置部分
# ==========================================
# 获取当前脚本所在目录的上一级目录的上一级目录（即项目根目录）
# 假设脚本位于 D:\projects\classes\week10\week10\scripts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据文件夹路径: D:\projects\classes\week10\week10\data\me
DATA_DIR = os.path.join(BASE_DIR, 'data', 'me')
# 结果文件保存路径
RESULT_PATH = os.path.join(DATA_DIR, 'result.csv')

# 相似度阈值 (0.0 - 1.0)，越高越严格
SIMILARITY_THRESHOLD = 0.2

# ------------------------------------------
# 实体抽取配置 (LLM vs Jieba)
# ------------------------------------------
# 是否使用大模型进行实体抽取 (True: 使用 LLM, False: 使用 Jieba)
USE_LLM_EXTRACTION = True 

# LLM 配置 (仅当 USE_LLM_EXTRACTION = True 时生效)
LLM_API_KEY = "sk-lyrzoilwprfhhpeddrmoqzbvwlxnqzezjphlqqsvzxjbebra"  # 请替换为你的 API Key
LLM_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "Pro/zai-org/GLM-4.7"

# 初始化 OpenAI 客户端
llm_client = None
if USE_LLM_EXTRACTION and OpenAI:
    llm_client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

# ------------------------------------------
# 流程控制配置
# ------------------------------------------
# 是否跳过去重步骤，直接基于现有的 result.csv 进行实体抽取
# 如果设为 True，请确保 result.csv 已经存在
ONLY_EXTRACT_ENTITIES = True

# LLM 抽取时的自动保存间隔 (行数)
# 大模型处理较慢，定期保存可以防止意外中断导致前功尽弃
LLM_SAVE_INTERVAL = 10

def clean_text(text):
    """
    简单的文本清洗：去除标点符号和空格，统一转小写
    """
    if not isinstance(text, str):
        return str(text)
    # 移除特殊字符，只保留中文、字母、数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text.lower()

def similarity(s1, s2):
    """
    计算两个字符串的相似度 (0.0 - 1.0)
    使用 difflib.SequenceMatcher
    """
    return SequenceMatcher(None, s1, s2).ratio()

def extract_entities_llm(text):
    """
    使用大模型抽取医学实体
    """
    if not llm_client:
        return ""
    
    try:
        print('调用大模型')
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个医学助手。请从用户的文本中提取医学相关的实体词（如疾病、症状、药物、解剖部位等）。仅输出实体词，用空格分隔，不要包含其他内容。"},
                {"role": "user", "content": text}
            ],
            temperature=0.1, # 低温度以获得更确定的结果
            max_tokens=100
        )
        print('获取大模型结果')
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""

def extract_entities(row):
    """
    从 title, ask, answer 中抽取医学实体词
    支持 Jieba 和 LLM 两种模式
    """
    # 组合文本
    t = str(row.get('title', ''))
    q = str(row.get('ask', ''))
    a = str(row.get('answer', ''))
    full_text = f"{t} {q} {a}"

    # ------------------------------------------
    # 互斥逻辑：根据开关选择一种抽取方式
    # ------------------------------------------
    if USE_LLM_EXTRACTION:
        # 模式 1: 使用大模型抽取
        return extract_entities_llm(full_text)
    else:
        # 模式 2: 使用 Jieba 抽取 (默认)
        if jieba is None:
            return ""
        
        # 提取关键词
        # 为了提高速度，暂时不使用 POS 过滤 (allowPOS)。
        # 如果开启 allowPOS=('n', 'nz', 'vn')，会调用 posseg.cut，速度较慢 (Windows下不支持并行)。
        # TF-IDF 默认会过滤停用词，提取出的高权重词通常也是实体或重要概念。
        keywords = jieba.analyse.extract_tags(
            full_text, 
            topK=10, 
            withWeight=False, 
            allowPOS=()
        )
        
        # 过滤纯数字和单字（可选，视需求而定）
        final_keywords = [w for w in keywords if not w.replace('.', '', 1).isdigit()]
        
        return " ".join(final_keywords)

def process_data():
    """
    数据处理主流程：
    1. 读取或生成清洗后的数据 (deduplication)
    2. 抽取医学实体 (entity extraction)
    """
    print("="*50)
    print(f"开始数据处理任务")
    print(f"数据目录: {DATA_DIR}")
    print("="*50)
    
    all_data_clean = None
    output_path = RESULT_PATH

    # ------------------------------------------
    # Phase 1: 获取基础数据 (读取现有 result.csv 或 重新去重)
    # ------------------------------------------
    if ONLY_EXTRACT_ENTITIES and os.path.exists(RESULT_PATH):
        print(f"\n[模式: 仅抽取实体]")
        print(f"跳过去重步骤，直接读取现有文件: {RESULT_PATH}")
        try:
            all_data_clean = pd.read_csv(RESULT_PATH, encoding='utf-8-sig')
            # 确保 entities 列存在
            if 'entities' not in all_data_clean.columns:
                all_data_clean['entities'] = ""
            # 将 NaN 填充为空字符串，避免后续处理报错
            all_data_clean['entities'] = all_data_clean['entities'].fillna("")
            print(f"读取成功，共 {len(all_data_clean)} 行")
        except Exception as e:
            print(f"读取文件失败: {e}")
            return
    else:
        print(f"\n[模式: 全量处理 (合并 + 去重)]")
        # 1. 获取所有待处理的 CSV 文件
        if not os.path.exists(DATA_DIR):
            print(f"错误: 目录 {DATA_DIR} 不存在")
            return

        all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        input_files = [f for f in all_files if os.path.basename(f).lower() != 'result.csv' and 'result_' not in os.path.basename(f).lower()]
        
        if not input_files:
            print("未找到待处理的 CSV 文件")
            return

        print(f"找到 {len(input_files)} 个待处理文件:")
        for f in input_files:
            print(f" - {os.path.basename(f)}")

        # 2. 读取并合并
        df_list = []
        for f in input_files:
            print(f"\n正在读取文件: {os.path.basename(f)} ...")
            try:
                # 尝试常见编码
                try:
                    df = pd.read_csv(f, encoding='gb18030')
                except UnicodeDecodeError:
                    df = pd.read_csv(f, encoding='utf-8')
                
                print(f" -> 读取成功，行数: {len(df)}")
                df_list.append(df)
            except Exception as e:
                print(f" -> 读取失败: {e}")

        if not df_list:
            print("没有成功读取任何数据")
            return

        print("-" * 30)
        all_data = pd.concat(df_list, ignore_index=True)
        print(f"所有文件合并完成\n原始总行数: {len(all_data)}")

        # 3. 数据去重
        print("开始数据去重流程...")
        
        # Step 1: 精确去重
        print(" [Step 1] 执行精确去重 (Exact Match)...")
        before_exact = len(all_data)
        all_data.drop_duplicates(subset=['title'], keep='first', inplace=True)
        after_exact = len(all_data)
        print(f" -> 精确去重移除: {before_exact - after_exact} 行")
        print(f" -> 剩余: {after_exact} 行")

        # Step 2: 模糊去重
        print(f" [Step 2] 执行模糊去重 (Fuzzy Match, Threshold={SIMILARITY_THRESHOLD})...")
        
        # 生成 clean_title 用于比较
        print(" -> 生成清洗后的 Title 列...")
        all_data['clean_title'] = all_data['title'].apply(clean_text)
        
        # 排序：让相似的标题尽可能相邻
        print(" -> 正在排序...")
        all_data.sort_values(by='clean_title', inplace=True)
        
        # 重置索引
        all_data.reset_index(drop=True, inplace=True)
        
        keep_mask = [True] * len(all_data)
        remove_count = 0
        
        titles = all_data['clean_title'].tolist()
        n = len(titles)
        
        print(f" -> 开始遍历 {n} 条数据进行相似度比对...")
        
        # 只需要遍历一次，比较相邻的项
        # 窗口大小可以设为 2 (即只比较 i 和 i+1)
        # 如果需要更严格，可以比较 i 与 i+1, i+2 ... 
        # 这里为了速度，只比较相邻
        for i in tqdm(range(n - 1), desc="Fuzzy Deduplication"):
            if not keep_mask[i]:
                continue
                
            current_title = titles[i]
            next_title = titles[i+1]
            
            # 简单长度过滤：如果长度差异太大，肯定不相似 (可选优化)
            if abs(len(current_title) - len(next_title)) > 5 and len(current_title) > 10:
                continue

            sim = similarity(current_title, next_title)
            if sim >= SIMILARITY_THRESHOLD:
                # 标记下一条为删除 (保留当前条)
                keep_mask[i+1] = False
                remove_count += 1
        
        print(f" -> 模糊去重标记移除: {remove_count} 行")
        
        all_data_clean = all_data[keep_mask].copy()
        
        # 清理临时列
        if 'clean_title' in all_data_clean.columns:
            all_data_clean.drop(columns=['clean_title'], inplace=True)

        print(f"\n全部去重完成！\n -> 总移除重复行数: {len(df_list[0]) + (len(all_data) - len(all_data_clean)) if False else (before_exact - len(all_data_clean))}") # 简化计算
        print(f" -> 最终有效行数: {len(all_data_clean)}")

        # ------------------------------------------
        # 保存中间结果 (Checkpoint)
        # ------------------------------------------
        # 检查是否可写
        try:
            if os.path.exists(output_path):
                with open(output_path, 'a'): pass
        except PermissionError:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(DATA_DIR, f'result_{timestamp}.csv')
            print(f"警告: {RESULT_PATH} 被占用，将保存到: {output_path}")

        print("\n" + "-"*30)
        print(f"正在保存去重后的中间结果到: {output_path}")
        try:
            all_data_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"保存成功！(Checkpoint 1)")
        except Exception as e:
            print(f"保存失败: {e}")
            return

    # ------------------------------------------
    # Phase 2: 抽取医学实体词 (Keyword Extraction)
    # ------------------------------------------
    if jieba or USE_LLM_EXTRACTION:
        print("\n" + "-"*30)
        mode = "LLM" if USE_LLM_EXTRACTION else "Jieba TF-IDF"
        print(f"开始抽取医学实体词 (模式: {mode})...")
        
        # 如果是 LLM 模式，采用循环 + 定期保存策略
        if USE_LLM_EXTRACTION:
            print(f"提示: LLM 模式下，将每隔 {LLM_SAVE_INTERVAL} 条保存一次进度。")
            
            # 确保 entities 列存在
            if 'llm_entities' not in all_data_clean.columns:
                all_data_clean['llm_entities'] = ""
            
            # 统计待处理行数
            # 假设空字符串或 NaN 需要处理
            # 注意：pandas 读取空csv单元格可能为 NaN
            mask_todo = (all_data_clean['llm_entities'].isna()) | (all_data_clean['llm_entities'].astype(str).str.strip() == "")
            todo_indices = all_data_clean[mask_todo].index
            
            total_todo = len(todo_indices)
            print(f"待处理行数: {total_todo} / {len(all_data_clean)}")
            
            if total_todo == 0:
                print("所有数据均已包含实体，无需处理。")
            else:
                count = 0
                # 使用 tqdm 显示进度
                pbar = tqdm(todo_indices, desc="LLM Extracting")
                for idx in pbar:
                    row = all_data_clean.loc[idx]
                    entities = extract_entities(row)
                    all_data_clean.at[idx, 'llm_entities'] = entities
                    
                    count += 1
                    
                    # 定期保存
                    if count % LLM_SAVE_INTERVAL == 0:
                        all_data_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
                        pbar.set_postfix({"Saved": count})
                
                # 循环结束后最后保存一次
                all_data_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"LLM 抽取完成，结果已保存到: {output_path}")

        else:
            # Jieba 模式：速度快，直接 apply 即可
            tqdm.pandas(desc="Jieba Extracting")
            all_data_clean['entities'] = all_data_clean.progress_apply(extract_entities, axis=1)
            
            print("\n" + "-"*30)
            print(f"正在保存最终结果到: {output_path}")
            all_data_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"保存成功！")

        print("实体抽取完成！预览前5行实体:")
        if 'entities' in all_data_clean.columns:
            print(all_data_clean[['title', 'entities']].head().to_string())
            
    else:
        print("\n警告: 未安装 jieba 且未启用 LLM，跳过实体抽取步骤。")

    print("==================================================")
    print("处理结束")
    print("==================================================")

if __name__ == "__main__":
    process_data()
