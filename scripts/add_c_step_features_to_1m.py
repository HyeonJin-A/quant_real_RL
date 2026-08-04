import os
import time
import glob
import pandas as pd
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "candle_data"))

def calculate_c_step_features(df):
    high_1m = df["high"]
    low_1m = df["low"]
    close_1m = df["close"]
    volume_1m = df["volume"]
    
    # Rolling 5m variables on 1m data
    high_roll = high_1m.rolling(5).max()
    low_roll = low_1m.rolling(5).min()
    close_roll_prev = close_1m.shift(5)
    vol_roll = volume_1m.rolling(5).sum()
    
    # 1 & 2. adx_5m and atr_5m
    tr1 = high_roll - low_roll
    tr2 = (high_roll - close_roll_prev).abs()
    tr3 = (low_roll - close_roll_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up = high_roll - high_roll.shift(5)
    down = low_roll.shift(5) - low_roll
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    def rma(series, window):
        alpha = 1 / window
        return pd.Series(series).ewm(alpha=alpha, adjust=False).mean()
    
    period = 70  # 14 * 5
    atr_5m = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / atr_5m
    minus_di = 100 * rma(minus_dm, period) / atr_5m
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx_5m = rma(dx, period)
    
    # 3. volatility_ratio
    range_roll = high_roll - low_roll
    ambient_vol = range_roll.shift(5).rolling(50, min_periods=1).mean()
    denom_vol = np.where(ambient_vol == 0, 1.0, ambient_vol)
    volatility_ratio = np.where(ambient_vol > 0, range_roll / denom_vol, 1.0)
    
    # 4. relative_volume_strength
    def rolling_volume_strength(volumes, strength_period):
        s = pd.Series(volumes)
        roll_min = s.rolling(strength_period, min_periods=1).min().values
        roll_max = s.rolling(strength_period, min_periods=1).max().values
        denom = roll_max - roll_min
        denom_safe = np.where(denom == 0, 1.0, denom)
        strength = np.where(denom > 0, (volumes - roll_min) / denom_safe, 0.0)
        return np.clip(strength, 0.0, 1.0)
    
    relative_volume_strength = rolling_volume_strength(vol_roll.values, 2500)
    
    # Assign and round
    df["5m_adx"] = adx_5m.fillna(0.0).round(2)
    df["5m_atr"] = atr_5m.fillna(0.0).round(2)
    df["5m_vol_ratio"] = np.round(volatility_ratio, 2)
    df["5m_vol_str"] = np.round(relative_volume_strength, 4)
    
    return df

def process_file(file_path):
    print(f"Processing: {file_path}")
    t0 = time.time()
    
    # Read CSV
    df = pd.read_csv(file_path)
    print(f"  Rows loaded: {len(df):,}")
    
    # Sort ascending for rolling calculations
    df_asc = df.sort_values("ts", ascending=True).reset_index(drop=True)
    
    # Calculate features
    print("  Calculating C step features (5m_adx, 5m_atr, 5m_vol_ratio, 5m_vol_str)...")
    df_asc = calculate_c_step_features(df_asc)
    
    # Restore original descending order
    df_desc = df_asc.sort_values("ts", ascending=False).reset_index(drop=True)
    
    # Output file path
    out_path = file_path.replace("_1m.csv", "_1m_full_features.csv")
    df_desc.to_csv(out_path, index=False)
    
    elapsed = time.time() - t0
    print(f"  Saved to: {out_path} ({elapsed:.2f}s)")
    print("  Head preview:")
    cols_to_show = ["ts", "close"]
    if "5m_rsi" in df_desc.columns:
        cols_to_show.append("5m_rsi")
    cols_to_show.extend(["5m_adx", "5m_atr", "5m_vol_ratio", "5m_vol_str"])
    print(df_desc[cols_to_show].head(5))
    print()

def main():
    target_files = [
        os.path.join(DATA_DIR, "BTC-USDT-SWAP_1m.csv"),
        os.path.join(DATA_DIR, "ETH-USDT-SWAP_1m.csv"),
    ]
    files = [f for f in target_files if os.path.exists(f)]
    
    if not files:
        print(f"No *_1m.csv files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} files to process in {DATA_DIR}:\n" + "\n".join(files) + "\n")
    for file_path in files:
        process_file(file_path)

if __name__ == "__main__":
    main()
