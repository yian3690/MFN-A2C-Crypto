from binance.client import Client
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import datetime

api_key = ''
api_sec = ''

client = Client(api_key, api_sec)
symbol = ['BNBUSDT', 'BTCUSDT', 'LTCUSDT', 'ETHUSDT']
symbol_2 = ['BTCUSDT']


def merge_columns_from_files(file_column_map, output_file):
    combined_df = pd.DataFrame()
    column_counter = 0

    for file, columns in file_column_map.items():
        try:
            df = pd.read_csv(file)

            if isinstance(columns, list):
                selected_columns = df[columns]
            else:
                selected_columns = df[[columns]]

            selected_columns = selected_columns.copy()
            selected_columns.columns = [f"{col}{column_counter}" for col in selected_columns.columns]
            column_counter += 1

            combined_df = pd.concat([combined_df, selected_columns], axis=1)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    combined_df.to_csv(output_file, index=False)
    print(f"合併完成，結果已保存到 {output_file}")


def calculate_price_change_rate(input_file, output_file):
    try:

        df = pd.read_csv(input_file)

        # df = df[columns]

        df_pct_change = df.pct_change()*100
        df_pct_change = df_pct_change.round(3)

        #df_pct_change = df_pct_change.iloc[1:]

        # df_pct_change.loc[len(df_pct_change)] = pd.Series()
        # df_pct_change.shift()

        df_pct_change.to_csv(output_file, index=False)
        print(f"計算完成，結果已保存到 {output_file}")

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")


def dumping():

    for sym in symbol:
        klines = client.get_historical_klines(
            symbol=sym,
            interval=Client.KLINE_INTERVAL_4HOUR,
            start_str="1 Jan, 2018",
            end_str="1 Sep, 2025"
            # start_str="1 Mar, 2025",
            # end_str="22 Apr, 2025"
        )

        df_M = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                                             'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                                             'Taker Buy Quote Asset Volume', 'Ignore'])

        columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']

        for col in columns_to_convert:
            df_M[col] = df_M[col].astype(float)
        df_M.to_csv(sym+'.csv')
        print(df_M)


    file_column_map = {
        'BTCUSDT.csv': ['Open', 'High', 'Low', 'Close'],
        'ETHUSDT.csv': ['Open', 'High', 'Low', 'Close'],
        'LTCUSDT.csv': ['Open', 'High', 'Low', 'Close'],
        'BNBUSDT.csv': ['Open', 'High', 'Low', 'Close']
    }
    file_column_map2 = {
        'BTCUSDT.csv': ['Close'],
        'ETHUSDT.csv': ['Close'],
        'LTCUSDT.csv': ['Close'],
        'BNBUSDT.csv': ['Close']
    }

    merge_columns_from_files(file_column_map, 'merged_output.csv')
    merge_columns_from_files(file_column_map2, 'avg.csv')
    calculate_price_change_rate('merged_output.csv', 'pct_change_output.csv')


def cal_ta():
    df = pd.read_csv('merged_output.csv')
    num_stocks = len([col for col in df.columns if col.startswith("Close")])
    result = pd.DataFrame()
    for i in range(num_stocks):
        prefix = f"{i}"

        o = df[f'Open{prefix}']
        h = df[f'High{prefix}']
        l = df[f'Low{prefix}']
        c = df[f'Close{prefix}']

        temp = pd.DataFrame({
            'open': o,
            'high': h,
            'low': l,
            'close': c
        })

        result[f'sma_{prefix}'] = ta.sma(temp['close'], length=20).pct_change()*100
        result[f'ema_{prefix}'] = ta.ema(temp['close'], length=20).pct_change()*100
        result[f'rsi_{prefix}'] = (ta.rsi(temp['close'], length=14)-50)*0.1

        macd = ta.macd(temp['close'])
        result[f'macd_{prefix}'] = macd[f"MACD_12_26_9"].pct_change()*100

    result.to_csv("ta_test.csv", index=False)
    print(result)


def preprocessing():
    df1 = pd.read_csv("merged_output.csv")
    df2 = pd.read_csv("ta_test.csv")
    df3 = pd.read_csv("pct_change_output.csv")
    df4 = pd.read_csv("avg.csv")

    df1 = df1.iloc[26:]
    df2 = df2.iloc[26:]
    df3 = df3.iloc[26:]
    df4 = df4.iloc[26:]

    df1.to_csv("merged_output.csv", index=False)
    df2.to_csv("ta_test.csv", index=False)
    df3.to_csv("pct_change_output.csv", index=False)
    df4['mean'] = np.mean(df4.values, axis=1)
    df4['val'] = df4['mean']*2.48115
    df4.to_csv("avg.csv", index=False)


def split_csv_by_tail(input_csv,
                      test_rows=1080,
                      train_out_csv=None,
                      test_out_csv=None,
                      keep_index=False):

    df = pd.read_csv(input_csv)
    n = len(df)
    if test_rows < 0:
        raise ValueError("test_rows 必須為非負整數")

    if test_rows >= n:
        test_df = df.copy()
        train_df = df.iloc[0:0].copy()  # 空的 DataFrame，保留欄位
        print(f"警告：檔案僅有 {n} 筆資料，已將全部 {n} 筆當作測試集，訓練集為空。")
    else:
        split_idx = n - test_rows
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df  = df.iloc[split_idx:].reset_index(drop=True)
        print(f"總筆數: {n} → 訓練: {len(train_df)} 筆, 測試: {len(test_df)} 筆 (最後 {test_rows} 筆)")

    base, ext = os.path.splitext(os.path.basename(input_csv))
    if train_out_csv is None:
        train_out_csv = base + "_train" + ext
    if test_out_csv is None:
        test_out_csv = base + "_test" + ext

    train_df.to_csv(train_out_csv, index=keep_index)
    test_df.to_csv(test_out_csv, index=keep_index)
    print(f"已儲存：訓練 -> '{train_out_csv}'，測試 -> '{test_out_csv}'")

    return train_df, test_df


if __name__ == '__main__':
    dumping()
    cal_ta()
    preprocessing()
    split_csv_by_tail("merged_output.csv")
    split_csv_by_tail("ta_test.csv")
    split_csv_by_tail("pct_change_output.csv")
    split_csv_by_tail("avg.csv")

    df4 = pd.read_csv("avg_test.csv")
