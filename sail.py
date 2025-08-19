import argparse
import numpy as np
import ast


def combine_preds_vec(preds, eps=1e-12):
    """
    Compute the SAIL inner (non-sigmoided) combination for n predictions.
    Matches the mathematical definition:
    
      inner(p1,...,pn) = 1 -   sum_{1<=i<j<=n} (1-p_i)(1-p_j)
                           -------------------------------
                               sum_{k=1}^n p_k

    - preds may be a 1D array of length N (treated as a single example)
      or a 2D array of shape (B, N) for a batch of B examples.
    - Returns a scalar (if 1D input) or an array of shape (B,).
    - If sum_k p_k is numerically zero (<= eps) the function returns 0.0
      for that row (consistent with a safe numerical convention).
    """
    p = np.asarray(preds, dtype=np.float64)

    # normalize input shape to (B, N)
    squeezed = False
    if p.ndim == 0:
        raise ValueError("preds must be an array of length >= 1")
    if p.ndim == 1:
        p = p[np.newaxis, :]
        squeezed = True
    elif p.ndim > 2:
        raise ValueError("preds must be 1D or 2D (batch, features)")

    B, N = p.shape
    if N == 0:
        raise ValueError("No predictions provided (N == 0).")

    # N == 1: no pairwise terms, inner == p_1
    if N == 1:
        out = p[:, 0].astype(np.float64)
        return out[0] if squeezed else out

    # q_i = 1 - p_i
    q = 1.0 - p                     # shape (B, N)
    sum_q = np.sum(q, axis=1)       # shape (B,)
    sum_q2 = np.sum(q * q, axis=1)  # shape (B,)

    # sum_{i<j} q_i q_j = (sum_q^2 - sum_q2) / 2
    numer = (sum_q * sum_q - sum_q2) / 2.0   # shape (B,)

    denom = np.sum(p, axis=1)       # shape (B,)

    # safe division: if denom is near zero, set inner to 0.0
    safe = denom > eps
    inner = np.zeros_like(denom, dtype=np.float64)
    inner[safe] = 1.0 - (numer[safe] / (denom[safe] + eps))

    return inner[0] if squeezed else inner


def sigmoid_pred_vec(pf):
    """
    Numerically stable sigmoid-like mapping used in original code:
      x = 10*(pf - 0.5) and then logistic(x) with clipping to avoid overflow.
    Accepts scalar, 1D, or ND arrays and returns same shape.
    """
    pf = np.asarray(pf, dtype=np.float64)
    x = 10.0 * (pf - 0.5)
    # clip to avoid extreme exponentials
    x = np.clip(x, -10.0, 10.0)

    # stable computation
    out = np.empty_like(x)
    pos = x >= 0
    if np.any(pos):
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    if np.any(~pos):
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
    return out


def sail_preds_vec(preds, start=1):
    """
    Generalized SAIL prediction combiner.
    preds: array-like shape (B, N) or (N,)
    start: integer column index to start using predictions from (default 1
           to preserve previous behaviour that ignored column 0).
           Set start=0 to use all columns.
    Returns: sigmoid(combined) of shape (B,) or scalar if input was 1D.
    """
    arr = np.asarray(preds, dtype=np.float64)
    if arr.ndim == 1:
        p = arr[start:]
    else:
        if start < 0 or start > arr.shape[1]:
            raise IndexError("start index out of range for preds columns.")
        p = arr[:, start:]

    combined = combine_preds_vec(p)
    return sigmoid_pred_vec(combined)


def parse_predictions(pred_string):
    """Parse prediction string from command line argument."""
    try:
        # Try to parse as a Python literal (list/array)
        preds = ast.literal_eval(pred_string)
        return np.array(preds, dtype=np.float64)
    except (ValueError, SyntaxError):
        # Try to parse as comma-separated values
        try:
            preds = [float(x.strip()) for x in pred_string.split(',')]
            return np.array(preds, dtype=np.float64)
        except ValueError:
            raise ValueError(f"Could not parse predictions: {pred_string}")


def main():
    parser = argparse.ArgumentParser(description='Apply SAIL prediction combination')
    
    parser.add_argument('--preds', type=str, required=True,
                        help='Array of predictions. Can be provided as: '
                             '1) Python list format: "[0.1, 0.2, 0.3]" '
                             '2) Comma-separated values: "0.1, 0.2, 0.3" '
                             '3) For 2D array: "[[0.1, 0.2], [0.3, 0.4]]"')
    parser.add_argument('--start', type=int, default=1,
                        help='Column index to start using predictions from (default: 1)')
    args = parser.parse_args()
    
    try:
        # Parse predictions
        predictions = parse_predictions(args.preds)
        
        # Apply SAIL combination
        result = sail_preds_vec(predictions, start=args.start)
        
        # Print result
        print(f"Input predictions: {predictions}")
        print(f"Start index: {args.start}")
        print(f"SAIL combined result: {result}")
        
        # Also print individual components for debugging
        if predictions.ndim == 1:
            p_subset = predictions[args.start:]
        else:
            p_subset = predictions[:, args.start:]
        
        combined_inner = combine_preds_vec(p_subset)
        print(f"Inner combination (before sigmoid): {combined_inner}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
