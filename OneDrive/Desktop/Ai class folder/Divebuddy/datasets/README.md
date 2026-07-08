# Datasets — future fine-tuning, not used by the current MVP

The MVP's detection stage runs on `Orange-OpenSource/marine-detect`'s
pretrained YOLOv8 checkpoints (18 classes total — see
`backend/app/rarity_map.json` for the full list), so no dataset download or
training was needed to get a working detector.

The two datasets below are the path to expanding species coverage past those
18 classes later, once real dive footage exposes what's missing:

- **FishNet** — https://github.com/faixan-khan/FishNet
  Large hierarchical fish dataset (taxonomy: order/family/genus/species).
  Best for teaching the model finer-grained fish species distinctions than
  marine-detect's family-level labels (e.g. Serranidae, Lutjanidae).

- **LILA Community Fish Dataset** — https://lila.science/datasets/community-fish-detection-dataset/
  Built from community-collected underwater footage rather than lab
  conditions — closer to what DiveBuddy actually processes, so more useful
  as a *domain-adaptation* fine-tuning set than as a source of new class
  labels.

## Why not fetched now

Both are large enough that a full download would eat most of this machine's
free disk headroom for no immediate benefit — the pretrained detector already
works. When it's time to fine-tune: pull a small stratified sample first
(not the full corpus), label any DiveBuddy-specific gaps with
[CVAT](https://www.cvat.ai/) or [Roboflow](https://universe.roboflow.com/),
and fine-tune `FishInv.pt`/`MegaFauna.pt` rather than training from scratch.
