"""Phase 5 · The risk indicator: from the scores of the two networks and the rules to 0-100.

Three steps, each with numbers measured on the data and nothing chosen by hand:

1. The score of the classifier is the plain mean of the front and the back networks.
2. Calibration turns that score into the probability that the ID is forged for a customer of
   the labelled population (where forgeries are about 1.4%, not the 7% of the training sample).
   It is fitted on scores given by models that never saw the person.
3. Each rule that FAILS multiplies the odds by its weight. The weight of a rule is the smallest
   likelihood ratio the data supports: the lowest plausible share of forged IDs that fail it
   divided by the highest plausible share of genuine IDs that fail it. A rule that has not
   failed often enough to tell weighs 1 (it changes nothing). PASS and NOT_EVALUABLE never
   lower or raise the risk.

The indicator is 100 times the resulting probability.
"""
import numpy as np

EPS = 1e-6
Z_95 = 1.645  # one-sided 95%


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, dtype=float), -50, 50)))


def fit_platt(scores, labels, weights=None, steps: int = 100) -> tuple[float, float]:
    """Logistic calibration: probability = sigmoid(slope * logit(score) + intercept). Two
    numbers, fitted by weighted maximum likelihood (Newton steps)."""
    x, y = logit(scores), np.asarray(labels, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])

    def loss(beta: np.ndarray) -> float:
        z = design @ beta
        return float((w * (np.logaddexp(0, z) - y * z)).sum())

    beta = np.array([1.0, 0.0])
    current = loss(beta)
    for _ in range(steps):
        p = sigmoid(np.clip(design @ beta, -50, 50))
        gradient = design.T @ (w * (p - y))
        hessian = (design * (w * p * (1 - p))[:, None]).T @ design + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        size = 1.0
        while size > 1e-6 and loss(beta - size * step) > current:  # a full Newton step can overshoot
            size /= 2
        beta = beta - size * step
        previous, current = current, loss(beta)
        if previous - current < 1e-10 * max(1.0, abs(previous)):
            break
    return float(beta[0]), float(beta[1])


def apply_platt(params: tuple[float, float], scores) -> np.ndarray:
    return sigmoid(params[0] * logit(scores) + params[1])


def fit_isotonic(scores, labels, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic calibration: the best non-decreasing step function from score to probability
    (pool adjacent violators). Returns the scores and the probabilities of its steps."""
    scores, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(scores, kind="stable")
    blocks = []  # [weighted sum of labels, weight, sum of score * weight]
    for s, label, weight in zip(scores[order], y[order], w[order]):
        blocks.append([label * weight, weight, s * weight])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1] = [blocks[-1][0] + top[0], blocks[-1][1] + top[1], blocks[-1][2] + top[2]]
    return np.array([b[2] / b[1] for b in blocks]), np.array([b[0] / b[1] for b in blocks])


def apply_isotonic(curve: tuple[np.ndarray, np.ndarray], scores) -> np.ndarray:
    return np.interp(np.asarray(scores, dtype=float), curve[0], curve[1])


def wilson_bounds(events: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Lowest and highest plausible rate behind `events` out of `total` (Wilson score bounds)."""
    if total == 0:
        return 0.0, 1.0
    rate = events / total
    centre = (rate + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


def rule_weight(failed_forged: int, evaluated_forged: int, failed_genuine: int, evaluated_genuine: int) -> float:
    """How many times a FAIL of the rule multiplies the odds of forgery: the lowest plausible
    failure rate among forged IDs over the highest plausible one among genuine IDs, and never
    below 1. A rule without enough failures to tell gets 1."""
    low_forged, _ = wilson_bounds(failed_forged, evaluated_forged)
    _, high_genuine = wilson_bounds(failed_genuine, evaluated_genuine)
    return max(1.0, low_forged / high_genuine) if high_genuine > 0 else 1.0


def indicator(probability, factor=1.0) -> np.ndarray:
    """Risk 0-100: the calibrated probability with its odds multiplied by `factor` (the product
    of the weights of the rules that failed; 1 when none did)."""
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    odds = p / (1 - p) * np.asarray(factor, dtype=float)
    return 100 * odds / (1 + odds)
