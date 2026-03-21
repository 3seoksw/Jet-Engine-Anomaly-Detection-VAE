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


def process_test(dset_dir: str, dset_name) -> pd.DataFrame:
    data_file = f"{dset_dir}/test_{dset_name}.txt"
    df = process_cmapss(data_file)

    rul_file = f"{dset_dir}/RUL_{dset_name}.txt"
    ruls = pd.read_csv(rul_file, header=None)
    ruls.columns = ["final_rul"]
    ruls["unit"] = range(1, len(ruls) + 1)

    last_cycle_nums = df.groupby("unit")["cycle"].max().to_frame(name="last_cycle")
    df = df.merge(last_cycle_nums, on="unit", how="left")
    df = df.merge(ruls, on="unit", how="left")
    df["rul"] = df["final_rul"] + df["last_cycle"] - df["cycle"]
    df["health_idx"] = df["rul"] / (df["last_cycle"] + df["final_rul"])
    df["health_level"] = df["rul"].apply(_assign_health_level)
    df.drop(columns=["last_cycle", "final_rul"], inplace=True)

    return df


def main():
    dset_dir = "data/CMAPSSData"
    dfs = []
    for i in range(1, 5):
        dset_name = f"FD00{i}"
        filename = f"{dset_dir}/train_{dset_name}.txt"
        df = process_cmapss(filename)
        dfs.append(df)
    train_df = pd.concat(dfs)
    rul_max = train_df["rul"].max()
    print(rul_max)

    dfs = []
    for i in range(1, 3):
        dset_name = f"FD00{i}"
        df = process_test(dset_dir, dset_name)
        dfs.append(df)
    val_df = pd.concat(dfs)

    dfs = []
    for i in range(3, 5):
        dset_name = f"FD00{i}"
        df = process_test(dset_dir, dset_name)
        dfs.append(df)
    test_df = pd.concat(dfs)

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


if __name__ == "__main__":
    main()
