import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _assign_health_level(rul: int):
    if rul > 120:
        return 0  # healthy
    elif rul < 30:
        return 2  # fatal (anomaly)
    else:
        return 1  # intermediate (uncertain)


def process_cmapss(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=" ", header=None)
    df.dropna(axis=1, how="all", inplace=True)
    _, n_cols = df.shape

    base_feats = ["unit", "cycle"]

    ops_feats = []
    for i in range(1, 4):
        ops_feats.append(f"ops_setting_{i}")

    sensor_feats = []
    for i in range(1, 22):
        sensor_feats.append(f"sensor_{i}")

    features = base_feats + ops_feats + sensor_feats
    assert len(features) == n_cols

    engine_num = filename.split(".")[0][-1]
    df["engine"] = engine_num
    features.append("engine")
    df.columns = features

    df = assign_features(df)

    return df


def assign_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add additional features; RUL, HI, Health Level ("Healthy", "Intermediate", "Fatal")
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = max_cycle - df["cycle"]
    df["health_idx"] = 1 - df["cycle"] / max_cycle
    df["health_level"] = df["rul"].apply(_assign_health_level)
    return df


def main():
    dset_dir = "data/CMAPSSData"

    train_dfs = []
    for fd in ["FD001", "FD002"]:
        df = process_cmapss(f"{dset_dir}/train_{fd}.txt")
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = train_df[train_df["rul"] >= 30].reset_index(drop=True)

    val_df = process_cmapss(f"{dset_dir}/train_FD003.txt")
    test_df = process_cmapss(f"{dset_dir}/train_FD004.txt")

    input_features = []
    input_features += [f"ops_setting_{i}" for i in range(1, 4)]
    input_features += [f"sensor_{i}" for i in range(1, 22)]

    scaler = MinMaxScaler()
    train_df[input_features] = pd.DataFrame(
        scaler.fit_transform(train_df[input_features].astype("float32"))
    )
    val_df[input_features] = pd.DataFrame(
        scaler.transform(val_df[input_features].astype("float32"))
    )
    test_df[input_features] = pd.DataFrame(
        scaler.transform(test_df[input_features].astype("float32"))
    )

    train_df.to_csv("data/cmapss_train.csv", index=False)
    val_df.to_csv("data/cmapss_val.csv", index=False)
    test_df.to_csv("data/cmapss_test.csv", index=False)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_units = df.groupby(["engine", "unit"]).ngroups
        print(
            f"{name}: {len(df)} rows, {n_units} units, engines {sorted(df['engine'].unique())}"
        )


if __name__ == "__main__":
    main()
