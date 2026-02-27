def accuracy_score(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true: list, y_pred: list, labels: list) -> list:
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n  = len(labels)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1
    return cm


def _precision_recall_f1(y_true: list, y_pred: list, pos_label) -> tuple:
    tp = sum(t == pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    fp = sum(t != pos_label and p == pos_label for t, p in zip(y_true, y_pred))
    fn = sum(t == pos_label and p != pos_label for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def f1_score(y_true: list, y_pred: list,
             average: str = "macro", pos_label=None) -> float:
    if average == "binary":
        _, _, f1 = _precision_recall_f1(y_true, y_pred, pos_label)
        return f1
    labels = sorted(set(y_true))
    f1s    = [_precision_recall_f1(y_true, y_pred, lbl)[2] for lbl in labels]
    return sum(f1s) / len(f1s) if f1s else 0.0


def classification_report(y_true: list, y_pred: list, target_names: list) -> str:
    labels = sorted(set(y_true))
    lines  = []
    lines.append(f"{'':>15}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}")
    lines.append("")

    all_p, all_r, all_f = [], [], []
    for lbl, name in zip(labels, target_names):
        p, r, f = _precision_recall_f1(y_true, y_pred, lbl)
        support = sum(t == lbl for t in y_true)
        lines.append(f"  {name:>13}  {p:>9.4f}  {r:>9.4f}  {f:>9.4f}  {support:>9}")
        all_p.append(p); all_r.append(r); all_f.append(f)

    total   = len(y_true)
    macro_p = sum(all_p) / len(all_p)
    macro_r = sum(all_r) / len(all_r)
    macro_f = sum(all_f) / len(all_f)

    lines.append("")
    lines.append(f"  {'macro avg':>13}  {macro_p:>9.4f}  {macro_r:>9.4f}  {macro_f:>9.4f}  {total:>9}")
    lines.append(f"  {'accuracy':>13}  {'':>9}  {'':>9}  {accuracy_score(y_true, y_pred):>9.4f}  {total:>9}")
    return "\n".join(lines)
