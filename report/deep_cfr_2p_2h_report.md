# Deep CFR 2-Player Training Report

Date: 2026-03-06

## Model

- Strategy weights: `models/deep_cfr_2p_2h/strategy_net_final.pt`
- Checkpoint: `models/deep_cfr_2p_2h/deep_cfr_2p_00060000.pt`
- Training log: `models/deep_cfr_2p_2h/train.log`

## Training Configuration

- Players: 2
- Device: mps
- Iterations: 60000
- Hidden size: 64
- Train every: 200
- Train steps: 30
- Batch size: 512
- Checkpoint every: 5000
- Max checkpoints kept: 2
- Seed: 42
- Blinds: 50/100
- Starting stack: 10000

## Training Completion

Final log lines:

- `2026-03-06 09:31:26 | INFO | Deep CFR iter 60000 | adv_bufs=[1000000, 1000000] | strat_buf=2000000`
- `2026-03-06 09:31:27 | INFO | Strategy net trained | loss: 0.9093`
- `2026-03-06 09:37:34 | INFO | Checkpoint saved: models/deep_cfr_2p_2h/deep_cfr_2p_00060000.pt`
- `2026-03-06 09:37:34 | INFO | [60000/60000] | adv_bufs=[1000000, 1000000] | strat_buf=2000000`
- `2026-03-06 09:37:34 | SUCCESS | Done. Strategy network saved: models/deep_cfr_2p_2h/strategy_net_final.pt`

## Evaluation Results

Evaluation command target:

- Model: `models/deep_cfr_2p_2h/strategy_net_final.pt`
- Hands per matchup: 2000
- Seed: 42
- Device: mps

### Vs Call Bot

- Win rate: 27.6%
- BB/100: +436.40
- Std Dev: 57.53 BB
- 95% CI: ±2.52 BB/hand

### Vs Random Bot

- Win rate: 40.2%
- BB/100: +481.89
- Std Dev: 39.65 BB
- 95% CI: ±1.74 BB/hand

## Notes

- The Deep CFR evaluation path was updated to place the loaded strategy network on the requested device, which was required for MPS evaluation.
- This model is profitable against both simple baseline opponents in the current evaluation setup.
