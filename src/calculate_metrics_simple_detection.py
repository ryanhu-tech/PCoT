import argparse
import pandas as pd
import json
import logging
import os
from utils.utils import compute_metrics
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_prediction(prediction_str):
    """
    解析 'generated_pred' 欄位中的 JSON 字串，提取預測標籤。
    """
    # 處理空值或非字串的情況
    if not isinstance(prediction_str, str):
        return None
    
    try:
        # 嘗試解析 JSON 格式，例如: {"response": "real"}
        data = json.loads(prediction_str)
        response = data.get('response')
        # 處理預測結果可能是 'True'/'False' 的情況
        if isinstance(response, str):
            response_lower = response.lower()
            if response_lower == 'true':
                return 'real'
            if response_lower == 'false':
                return 'fake'
        return response
    except json.JSONDecodeError:
        # 如果解析失敗，嘗試直接從字串中尋找關鍵字
        prediction_str = prediction_str.lower()
        if 'real' in prediction_str:
            return 'real'
        if 'fake' in prediction_str:
            return 'fake'
        # 處理預測結果可能是 'True'/'False' 的情況
        if 'true' in prediction_str:
            return 'real'
        if 'false' in prediction_str:
            return 'fake'
        return None

def main():
    """
    主執行函式，讀取 CSV 並計算指標。
    """
    parser = argparse.ArgumentParser(description="Calculate F1 score and other metrics from a prediction CSV file.")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to the final.csv file.")
    parser.add_argument('-o', '--output', type=str, help="Optional: Path to save the output metrics CSV file.")
    args = parser.parse_args()

    logging.info(f"Reading data from: {args.file}")
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        logging.error(f"File not found: {args.file}")
        return

    # 提取真實標籤和預測標籤
    y_true = df['label'].tolist()
    y_pred_raw = df['generated_pred'].tolist()

    # 解析預測欄位
    y_pred = [parse_prediction(p) for p in y_pred_raw]

    # 過濾掉無法解析的預測
    valid_indices = [i for i, p in enumerate(y_pred) if p is not None]
    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]

    logging.info(f"Total rows: {len(df)}. Valid predictions found: {len(y_true_filtered)}.")

    # 計算指標
    logging.info("Calculating metrics...")
    metrics = compute_metrics(y_true_filtered, y_pred_filtered)
    report_dict = classification_report(y_true_filtered, y_pred_filtered, output_dict=True, digits=4)

    # 如果有指定輸出檔案，就儲存成 CSV
    if args.output:
        # 將 classification_report 的字典轉換為 DataFrame
        report_df = pd.DataFrame(report_dict).transpose()
        
        # 將額外的 f1_binary_fake 加入
        # 為了對齊格式，我們將它放在一個 Series 中
        f1_binary_series = pd.Series({'f1-score': metrics['f1_binary_fake']}, name="f1_binary_fake")
        
        # 合併兩個 DataFrame
        final_df = pd.concat([report_df, f1_binary_series.to_frame().T])

        # 確保輸出目錄存在
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        final_df.to_csv(args.output, float_format='%.4f')
        logging.info(f"Metrics saved to {args.output}")
    else:
        # 否則，像之前一樣印出結果
        print("\n--- Metrics Report ---")
        print(classification_report(y_true_filtered, y_pred_filtered, digits=4))
        print(f"F1 Score (Binary 'fake'): {metrics['f1_binary_fake']:.4f}")
        print("----------------------")

if __name__ == '__main__':
    main()