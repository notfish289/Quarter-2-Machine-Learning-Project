import pandas as pd


IN_FILE = "data/credit_card.csv"
TRAIN_FILE = "data/diabetes_train.csv"
TEST_FILE = "data/diabetes_test.csv"


TRAIN_SPLIT = 0.8
NUM_BINS = 3


def main() -> None:
    df = pd.read_csv(IN_FILE)

    class_col = df.iloc[:, -1]
    df = df[df.columns[:-1]]

    for label in df.columns:
        min_value = df[label].min() - 1
        max_value = df[label].max()

        bin_boundaries = [
            min_value + i * (max_value - min_value) / NUM_BINS
            for i in range(NUM_BINS + 1)
        ]
        df[f"{label}_discrete"] = pd.cut(df[label], bins=bin_boundaries)

    df = df.join(class_col)

    train_df = df[: int(len(df) * TRAIN_SPLIT)]

    test_df = df[int(len(df) * TRAIN_SPLIT) :].copy()
    for label in test_df.columns[:-1]:
        if not label.endswith("_discrete"):
            test_df.drop(label, axis=1, inplace=True)
    test_df.rename(
        columns={c: c.removesuffix("_discrete") for c in df.columns}, inplace=True
    )

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)


if __name__ == "__main__":
    main()
