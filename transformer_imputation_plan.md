# Transformer-Based Imputation Plan for Electricity Consumption Curves

## Current Performance & Targets (from interpretation.md)

**Baselines (loss computed only on holes, standardized data):**

- Zero (mean): MSE=0.946 → RMSE≈0.973 (strongest baseline)
- Historical (48-step): MSE≈1.005
- Linear interp: MSE≈1.227
- Forward-fill: MSE≈1.571

**Targets:**

- Minimum viable: beat strongest baseline (<0.946 MSE)
- Acceptable: MSE < 0.10 (RMSE < ~0.32 std)
- Good: MSE < 0.04 (RMSE < 0.20 std) → visually good reconstructions
- Current best models: ~0.22 Huber / 0.58 MSE (RMSE ~0.65-0.77) → room for significant improvement

LSTM-based models (v1-v8) have plateaued around these values despite varying layers (4-16), hidden sizes, loss functions (MSE vs Huber), and window sizes (144 vs 336). This suggests fundamental limitations in capturing long-range periodic dependencies (daily/weekly patterns in electricity usage).

## Why Switch to Transformer Encoder?

**Advantages over LSTM for this task:**

1. **Global attention**: Self-attention directly models relationships between any timesteps, excellent for periodic signals (24h=48 steps, 7d=336 steps).
2. **No vanishing gradients**: Better gradient flow over 336-step sequences.
3. **Parallel computation**: Faster training.
4. **Multi-scale feature learning**: Multi-head attention can capture both local smoothness and daily/weekly cycles when combined with cyclical time features.
5. **Mask awareness**: The input already includes a binary mask + holed values (set to 0), allowing the model to learn "when to ignore vs. reconstruct".

**Challenges:**

- Quadratic attention cost (336² is ~113k operations per head — still very manageable on GPU).
- Needs explicit positional encodings (no recurrence).
- Risk of overfitting without proper regularization (dropout, weight decay).

## Proposed Architecture

### Core Components

**1. Input Projection**

- Linear layer: `input_dim=8` → `d_model` (recommend 256)
- Input features: `[holed_consumption, mask, grid_features (mean/std?, hour_sin/cos, day_sin/cos)]`

**2. Positional Encoding**

- Sinusoidal (fixed) or learnable embeddings added to input.
- Critical for time-series order.

**3. Transformer Encoder**

```python
# Recommended config
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,
    nhead=8,
    dim_feedforward=1024,
    dropout=0.1,
    activation='gelu',
    batch_first=True,
    norm_first=True  # Pre-norm for stability
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
```

**4. Decoder Head Options (to experiment with):**

- **Option A (current style)**: Conv1D stack on encoder output (channels-first). Good for local smoothness.
- **Option B (residual)**: Concatenate original input features (`x`) to encoder output → Conv1D or Linear per-timestep. Helps model focus on _corrections_ rather than full reconstruction.
- **Option C**: Simple MLP head: `Linear(d_model, 64) -> GELU -> Linear(64, 1)` applied across time.
- **Option D**: Small Transformer decoder or cross-attention if we want autoregressive flavor (less necessary here).

**5. Output**

- Predict full sequence `(B, 336, 1)`, but compute loss **only on holes** using `hole_mask = 1 - mask`.

### Loss Function Recommendations

- **Primary**: `nn.HuberLoss(reduction='none', delta=1.0)` — robust to consumption spikes.
- **Advanced**: Add temporal smoothness regularization (L1 on consecutive differences, masked).
- **Optional**: Focal loss weighting (higher weight on larger gaps).

## Training Improvements (from strategy.md + new)

1. **Learning Rate Schedule**
   - Start with `lr=5e-4` or `1e-3` (higher than previous 1e-4).
   - Use `OneCycleLR` or `CosineAnnealingWarmRestarts`.
   - Keep ReduceLROnPlateau as fallback.

2. **Regularization**
   - Dropout 0.1-0.2 in encoder.
   - Weight decay 1e-5.
   - Gradient clipping (already present).

3. **Data & Batch**
   - Batch size 512+ (or accumulate_grad_batches).
   - `train_fraction=0.8` or full.
   - Consider varying gap lengths more aggressively during training.

4. **Monitoring**
   - Log separate metrics for different gap lengths if possible.
   - Visualize reconstructions periodically (add to notebook).
   - Track vs baselines using `calculate_baselines(dm)`.

## Experiment Plan (v9+)

Update the table in `interpretation.md` with these:

- **v9**: TransformerEncoder (layers=4, d_model=128, heads=4) + Conv1D decoder, Huber, window=336, lr=5e-4
- **v10**: Transformer (layers=6, d_model=256, heads=8) + Residual Concat + Conv, Huber
- **v11**: Same as v10 but with smoothness loss term
- **v12**: Smaller model (d_model=192, layers=4) with higher LR + OneCycle
- **v13**: Test window=144 vs 336 with Transformer

**Success criteria per run:**

- Val loss < 0.15 (significant improvement)
- Val loss < 0.10 (target)
- RMSE < 0.32 std on holes

## Implementation Steps

1. **Create reusable components** (in notebook or extract to .py):
   - `PositionalEncoding` class.
   - New `TransformerImputerModel` class inheriting from `EnergyImputerModel` or standalone.

2. **Update forward pass** to handle positional encoding + transformer.

3. **Adjust decoder input channels** based on chosen architecture.

4. **Test baseline calculation** still works.

5. **Iterate**:
   - Train short runs (10-20 epochs) to compare configs.
   - Use TensorBoard for loss curves and LR.
   - Add validation visualizations (plot true vs predicted for sample holes).

6. **Next steps after success**:
   - Inference on real test holes.
   - Inverse scaling back to Watts.
   - Submit to challenge.

## Additional Ideas to Explore Later

- **Time-series specific models**: Informer, Autoformer, or PatchTST (if performance plateaus).
- **Hybrid**: LSTM + Attention or Conv + Transformer (S4, Mamba if available).
- **Pretraining**: Self-supervised on full (non-holed) data.
- **Per-meter clustering**: Group similar consumption patterns.
- **Uncertainty estimation**: Predict mean + variance.

This plan builds directly on existing code (DataModule, Dataset, loss masking, scaling) while addressing the core limitations of the LSTM approach. We should be able to achieve MSE < 0.10 with proper tuning.

Start by implementing the base TransformerEncoder version and compare to v6/v5.
