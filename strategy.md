# Set of strategies to reduce the error value and get better results in prediction

Note: we need to stablish a benchmark to known how the Criterion used is translated in a good value to reach in our training, what is good value for MSE and HuberLoss.

## 1. Increase the Learning Rate (with a warmup)

Why: A learning rate of 1e-4 is quite conservative for AdamW. If the deltas are shrinking fast, the optimizer is taking tiny steps and getting stuck in local minima too early. Action:

- Bump the base LR to 5e-4 or even 1e-3.
- Crucially, use a learning rate warmup. AdamW can destabilize in the first few epochs if the LR is high. PyTorch Lightning supports OneCycleLR or CosineAnnealingLR easily. Code tweak in configure_optimizers:

```python
optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=self.hparams.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3,
    total_steps=self.trainer.estimated_stepping_batches # Requires passing total steps
)
# Note: OneCycleLR needs interval='step' in Lightning
```

## 2. Implement "Teacher Forcing" via Residual Input Injection

Why: The decoder has to reconstruct 336 points from scratch based purely on the encoder's internal representations. In many sequence tasks, passing the original known inputs directly into the decoder layers alongside the encoder features acts like a cheat code, allowing the network to focus only on predicting the gaps rather than perfectly memorizing the parts it already knows. Action:

- Since you have a 1D Conv decoder, concatenate the original 8-dimensional input features (which contain time-of-day, day-of-week, etc.) to the encoder's output before it goes into the Conv decoder. Code tweak in forward:

```python
def forward(self, x):
    output, _ = self.encoder(x)           # (batch, 336, 256)

    # x shape is (batch, 336, 8)
    # Concatenate the original input features with the encoder hidden states
    combined = torch.cat([output, x], dim=-1) # (batch, 336, 264)

    combined = combined.transpose(1, 2)   # (batch, 264, 336)
    decoded = self.decoder(combined)      # (batch, 1, 336)
    return decoded.transpose(1, 2)
```

_(You will need to increase the first Conv1d in_channels from 256 to 256 + 8 = 264)._

## 3. Change the Loss Function to emphasize the "Holes"

Why: Your current loss computes the MSE over the missing data only. But early in training, predicting the non-missing data correctly builds excellent representations. Conversely, predicting the sharp spikes in electricity data is heavily penalized by pure MSE. Action:

- Add a penalty for temporal smoothness (e.g., the difference between step t and step t -1)
- Use Huber Loss (Smooth L1) instead of strict MSE. Electricity data often has heavy outliers (spikes in consumption). Pure MSE squares these errors, forcing the model to fixate on outliers and ignoring the general curve shape, which slows down broad convergence. Code tweak: Change `self.criterion = nn.MSELoss(reduction='none')` to `self.criterion = nn.HuberLoss(reduction='none', delta=1.0)`.

## 4. Increase batch_size (or accumulate_grad_batches)

Why: You reduced effective batch size from 512 to 256 (batch=256, accum=1) or 512 (batch=256, accum=2). Larger batches provide much smoother, more accurate gradient estimates, allowing you to safely use a higher learning rate (as suggested in point #1).

Action: If your GPU permits, increase `accumulate_grad_batches` to 4 or 8 to simulate a batch size of 1024. Combine this with the higher learning rate.

## 5. Expand the Conv1D Receptive Field (Dilated Convolutions)

Why: Your current Conv1D decoder sees about ~20 timesteps of context per output point. For electricity data, patterns repeat daily (48 timesteps at 30m resolution) and weekly. If the model can't "see" what happened 24 hours ago directly in the decoder, it learns slower. Action: Add dilation to the deeper Conv1D layers. This exponentially expands the receptive field without adding parameters. Code tweak in `decoder`:

```python
self.decoder = nn.Sequential(
    # ... previous layers ...
    nn.Conv1d(128, 64, kernel_size=5, padding=4, dilation=2), # Looks wider
    nn.GELU(),
    # ...
)
```

---

Implementation **3. Change the Loss Function to emphasize the "Holes"**

Here is the implementation of the change in the `_compute_masked_loss`:

```python
    def __init__(self, input_dim=8, hidden_dim=128, window_size=336,
                 num_layers=4, lr=1e-4, weight_decay=1e-5):
        # ... keep everything else the same ...

        # Change MSE to Huber
        self.criterion = nn.HuberLoss(reduction='none', delta=1.0)

    def _compute_masked_loss(self, outputs, y_batch, mask_batch, lambda_smooth=0.1):
        """Shared masked loss logic for training and validation."""
        # 1. Base reconstruction loss (now Huber)
        base_loss = self.criterion(outputs, y_batch)
        hole_mask = 1 - mask_batch
        denominator = hole_mask.sum()

        if denominator > 0:
            reconstruction_loss = (base_loss * hole_mask).sum() / denominator
        else:
            reconstruction_loss = base_loss.mean()

        # 2. Temporal Smoothness Penalty (Derivative loss)
        # Calculate differences between consecutive timesteps for both pred and target
        diff_outputs = outputs[:, 1:, :] - outputs[:, :-1, :]
        diff_targets = y_batch[:, 1:, :] - y_batch[:, :-1, :]

        # We also need a mask for the differences. If a hole spans t-1 to t, we penalize it.
        diff_mask = hole_mask[:, 1:, :] * hole_mask[:, :-1, :]
        diff_denominator = diff_mask.sum()

        if diff_denominator > 0:
            # We can just use L1 or MSE for the differences
            smoothness_loss = (torch.abs(diff_outputs - diff_targets) * diff_mask).sum() / diff_denominator
        else:
            smoothness_loss = torch.abs(diff_outputs - diff_targets).mean()

        # Combine them
        total_loss = reconstruction_loss + (lambda_smooth * smoothness_loss)
        return total_loss
```

# Some observations around the experiments performed

we are going to stop using LSTM, they have a fundamental limitation in the curve reconstruction and we are always getting similar results, independent of the the number of layers that we used, for this reason we are going to opt for a model that implements a transformer as memory mechanism
