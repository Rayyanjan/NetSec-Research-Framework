## Algorithm 1 — SlideFU Gradient Calibration Unlearning

```
Algorithm 1: SlideFU(state_global, W_buf, C_target, sizes)
─────────────────────────────────────────────────────────────────
Input:
    state_global  : current global model parameter vector (R^d)
    W_buf         : sliding-window store of per-client gradient
                    deltas, |W_buf[c]| ≤ W = 10 for each client c
    C_target      : set of client IDs to unlearn (|C_target| = m)
    sizes         : array of per-client local-dataset sizes
Output:
    state_unlearn : post-unlearning global parameter vector

 1: total_size  ← Σ_c sizes[c]                     // for FedAvg-weights
 2: correction  ← 0  ∈ R^d                         // initialise
 3: for each c ∈ C_target do
 4:     w_c ← sizes[c] / total_size                // FedAvg weight
 5:     for each Δ ∈ W_buf[c] do                   // Δ : R^d
 6:         correction ← correction + w_c · Δ      // O(W) accumulation
 7:     end for
 8: end for
 9: state_unlearn ← state_global − correction      // reverse contribution
10: state_unlearn ← FineTune(state_unlearn, 2 rounds, honest clients)
                                                   // brief recovery
11: return state_unlearn

Complexity:
    Time   : O(m · W · d)              // m targets × W history × d params
    Space  : O(n · W · d)              // sliding window across n clients
                                       //   (constant in number of rounds T)
    vs. Full Retrain : O(T · n · d) where T ≫ W
```

**Caption.** Algorithm 1 implements the SlideFU sliding-window gradient calibration step. The key insight is that the unlearning contribution of a target client *c* over a window of *W* recent rounds equals the FedAvg-weighted sum of its stored gradient deltas; subtracting that quantity from the current global state reverses *c*'s influence in O(W) time without touching contributions from honest clients. A brief 2-round honest-client fine-tune (line 10) restores any small drift introduced by the calibration.

---

## Algorithm 2 — UnlearnGuard PCA-Based Filtering

```
Algorithm 2: UnlearnGuard(W_buf, C_target, k, τ)
─────────────────────────────────────────────────────────────────
Input:
    W_buf      : sliding-window store of per-client gradient deltas
    C_target   : set of client IDs whose history is being filtered
                 (suspected adversarial unlearning A_2)
    k          : number of PCA components (default k = 8)
    τ          : trust threshold on cosine similarity (default τ = 0.65)
Output:
    W_filtered : sanitised window store with adversarial deltas
                 zeroed out
─────────────────────────────────────────────────────────────────
 1: H ← {hist[-1] : c ∉ C_target, hist = W_buf[c], hist ≠ ∅}
                                                      // most-recent
                                                      // honest deltas
 2: PCA_k ← FitPCA(H, k components)                   // R^d → R^k
 3: H_proj    ← PCA_k.transform(H)                    // R^|H|×k
 4: centroid  ← mean(H_proj, axis=0)                  // R^k
 5: ĉ         ← centroid / ‖centroid‖                 // unit vector

 6: W_filtered ← copy(W_buf)
 7: for each c ∈ C_target do
 8:     for each Δ ∈ W_filtered[c] (in order) do
 9:         v ← PCA_k.transform(Δ)                    // R^k
10:         v̂ ← v / ‖v‖
11:         cos_sim ← ⟨v̂, ĉ⟩                         // ∈ [−1, 1]
12:         if cos_sim < τ then                       // anomalous
13:             replace Δ with 0 ∈ R^d                // reject
14:         end if
15:     end for
16: end for
17: return W_filtered

Complexity:
    Time   : O(d · n + d · k + |C_target| · W · k)
             // PCA fit dominated by O(d·n) when n < d
    Space  : O(d · k)                  // PCA basis matrix
```

**Caption.** Algorithm 2 implements UnlearnGuard's PCA-based adversarial-update filter. Lines 1–5 build a low-dimensional "honest" reference direction from recent honest-client gradient deltas. Lines 7–16 then project each gradient delta in the target clients' window onto this PCA subspace and gate by cosine similarity; updates whose direction deviates more than τ = 0.65 from the honest centroid are rejected (replaced with zero). This neutralises the inflated, gradient-aligned vectors crafted by adversary A₂ (BadUnlearn).

---
