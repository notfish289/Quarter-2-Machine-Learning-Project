import numpy as np
import pandas as pd
import random
from scipy import stats  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore

random_seed = 33
np.random.seed(random_seed)
random.seed(random_seed)


TRAIN_FILE = "data/diabetes_train.csv"
TEST_FILE = "data/diabetes_test.csv"

K = 2


def calc_entropy(x: np.ndarray) -> float:
    p = np.sum(x) / x.size if x.size != 0 else 0

    if p in (0, 1):
        return 0

    return -p * np.log2(p)


def compute_distances(x: np.ndarray, y: np.ndarray, no_sqrt: bool = True) -> np.ndarray:
    """
    computes L2 distance between each row in x and y
    """
    # x^2 - 2xy + y^2
    # (x - y)^2
    distances = np.sum(x**2, 1, keepdims=True) - 2 * x @ y.T + np.sum(y**2, 1)

    if not no_sqrt:
        distances[distances < 0] = 0
        distances = np.sqrt(distances)

    return distances


def k_means_cluster(
    instances: np.ndarray, k: int = K
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    # starting centers of K random points from the dataset
    centers = instances[np.random.randint(instances.shape[0], size=(k,))]
    center_points: list[np.ndarray] = [np.empty(0) for _ in range(k)]
    center_point_idxs: list[np.ndarray] = [np.empty(0) for _ in range(k)]

    while True:
        # distances between each center and instances
        distances = compute_distances(centers, instances)
        # idx of the center corresponding to the min distance
        min_indices = np.argmin(distances, 0)

        centers_are_same = True

        for center_idx in range(k):
            new_center_point_idxs = np.where(min_indices == center_idx)[0]
            new_center_points = instances[new_center_point_idxs]
            new_center = (
                np.mean(new_center_points, 0)
                if new_center_points.size > 0
                else instances[np.random.randint(instances.shape[0])]
            )

            if (
                new_center_points.size > 0
                and not np.isclose(centers[center_idx], new_center).all()
            ):
                centers_are_same = False

            centers[center_idx] = new_center
            center_points[center_idx] = new_center_points
            center_point_idxs[center_idx] = new_center_point_idxs

        if centers_are_same:
            break

    return centers, center_points, center_point_idxs


def _build_decision_tree(
    discrete_df: pd.DataFrame,
    continuous_data: np.ndarray,
    class_values: np.ndarray,
    available_attrs: list[str],
    attr_unique_vals: list[np.ndarray],
) -> tuple[str, dict[str, tuple]] | int:
    if class_values.size == 0:
        return 0
    if not available_attrs or np.unique(class_values).size == 1:
        return int(stats.mode(class_values).mode)

    attribute_entropies = {}

    for idx, attribute in enumerate(available_attrs):
        full_attribute_entropy = 0

        attribute_col = continuous_data[:, idx]
        continuous_data = np.delete(continuous_data, idx, 1)
        for v in attr_unique_vals[idx]:
            continous_split_on_val = continuous_data[
                discrete_df[attribute].to_numpy() == v
            ]

            if continous_split_on_val.size == 0:
                continue

            *_, k_cluster_idxs = k_means_cluster(continous_split_on_val)

            full_clustered_entropy = 0

            for k_idxs in k_cluster_idxs:
                cluster_class_values = class_values[k_idxs]
                cluster_entropy = calc_entropy(cluster_class_values)

                full_clustered_entropy += (
                    cluster_entropy * k_idxs.size / continous_split_on_val.shape[0]
                )

            full_attribute_entropy += (
                full_clustered_entropy
                * continous_split_on_val.shape[0]
                / continuous_data.shape[0]
            )

        continuous_data = np.insert(continuous_data, idx, attribute_col, 1)

        attribute_entropies[attribute] = full_attribute_entropy

    best_attr = min(attribute_entropies, key=attribute_entropies.__getitem__)
    attr_idx = available_attrs.index(best_attr)
    new_available_attrs = [a for a in available_attrs if a != best_attr]
    new_attr_unique_vals = [v for i, v in enumerate(attr_unique_vals) if i != attr_idx]
    new_continuous_data = np.delete(continuous_data, attr_idx, 1)

    decision_tree: tuple[str, dict] = (best_attr, {})
    for unique_val in attr_unique_vals[attr_idx]:
        split_idxs = discrete_df[best_attr].to_numpy() == unique_val
        new_discrete_df = discrete_df.iloc[split_idxs]
        partitioned_continuous_data = new_continuous_data[split_idxs]
        new_class_values = class_values[split_idxs]

        decision_tree[1][unique_val] = _build_decision_tree(
            new_discrete_df,
            partitioned_continuous_data,
            new_class_values,
            new_available_attrs,
            new_attr_unique_vals,
        )

    return decision_tree


def build_decision_tree(df: pd.DataFrame) -> tuple[str, dict[str, tuple]]:
    class_values = df.iloc[:, -1].to_numpy()
    df.drop(df.columns[-1], axis=1, inplace=True)
    continuous_data = []

    for label in df.columns:
        if not label.endswith("_discrete"):
            continuous_data.append(np.expand_dims(df[label].to_numpy(), 1))
            df.drop(label, axis=1, inplace=True)

    np_continuous_data = np.concatenate(continuous_data, 1)
    df.rename(
        columns={c: c.removesuffix("_discrete") for c in df.columns}, inplace=True
    )
    attribute_unique_values = [df[c].unique() for c in df.columns]

    decision_tree = _build_decision_tree(
        df, np_continuous_data, class_values, [*df.columns], attribute_unique_values
    )

    if isinstance(decision_tree, int):
        raise ValueError()

    return decision_tree


def predict(
    decision_tree: tuple[str, dict[str, tuple]] | int,
    instance: np.ndarray,
    attrs: list[str],
) -> int:
    while True:
        if not isinstance(decision_tree, tuple):
            return decision_tree

        attr, result = decision_tree

        attr_value = instance[attrs.index(attr)]

        if attr_value not in result:
            return 0

        decision_tree = result[attr_value]


def load_train_for_evaluation() -> pd.DataFrame:
    train_df = pd.read_csv(TRAIN_FILE)
    for label in train_df.columns[:-1]:
        if not label.endswith("_discrete"):
            train_df.drop(label, axis=1, inplace=True)
    train_df.rename(
        columns={c: c.removesuffix("_discrete") for c in train_df.columns}, inplace=True
    )

    return train_df


def load_scikit_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = load_train_for_evaluation()
    test_df = pd.read_csv(TEST_FILE)

    train_data = train_df.to_numpy().astype(str)
    test_data = test_df.to_numpy().astype(str)

    X_train = train_data[:, :-1]

    X_train = np.array(
        [
            np.unique(X_train[:, col], return_inverse=True)[1]
            for col in range(X_train.shape[1])
        ]
    ).T
    Y_train = train_data[:, -1].astype(np.int32)

    X_test = test_data[:, :-1]
    X_test = np.array(
        [
            np.unique(X_test[:, col], return_inverse=True)[1]
            for col in range(X_test.shape[1])
        ]
    ).T
    Y_test = test_data[:, -1].astype(np.int32)

    return X_train, Y_train, X_test, Y_test


def print_metrics(
    model_name: str, type: str, preds: np.ndarray, labels: np.ndarray
) -> None:
    conf_matrix = confusion_matrix(labels, preds)
    TN, FP, FN, TP = conf_matrix.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"##### {model_name} {type} #####")
    print(f"Accuracy: {accuracy:.2%}")
    print("Confusion Matrix: ")
    print("\t Actual (1) Actual (0)")
    print(
        f"Pred (1) | {TP:0>3}\t      {FP:0>3}\nPred (0) | {FN:0>3}\t      {TN:0>3}"
    )
    print()


def main() -> None:
    train_df = load_train_for_evaluation()
    test_df = pd.read_csv(TEST_FILE)
    attrs = [*test_df.columns[:-1]]

    train_dataset = train_df.to_numpy()
    train_X = train_dataset[:, :-1]
    train_Y = train_dataset[:, -1].astype(np.int64)

    test_dataset = test_df.to_numpy()
    test_X = test_dataset[:, :-1]
    test_Y = test_dataset[:, -1].astype(np.int64)

    decision_tree = build_decision_tree(pd.read_csv(TRAIN_FILE))

    train_preds = np.array([predict(decision_tree, x, attrs) for x in train_X])
    test_preds = np.array([predict(decision_tree, x, attrs) for x in test_X])

    print_metrics("Clustered Decision Tree", "Train", train_preds, train_Y)
    print_metrics("Clustered Decision Tree", "Test", test_preds, test_Y)

    scikit_train_X, scikit_train_Y, scikit_test_X, scikit_test_Y = load_scikit_dataset()

    scikit_decision_tree = DecisionTreeClassifier()
    scikit_decision_tree.fit(scikit_train_X, scikit_train_Y)

    scikit_train_preds = scikit_decision_tree.predict(scikit_train_X)
    scikit_test_preds = scikit_decision_tree.predict(scikit_test_X)

    print_metrics("Regular Decision Tree", "Train", scikit_train_preds, scikit_train_Y)
    print_metrics("Regular Decision Tree", "Test", scikit_test_preds, scikit_test_Y)


if __name__ == "__main__":
    main()
