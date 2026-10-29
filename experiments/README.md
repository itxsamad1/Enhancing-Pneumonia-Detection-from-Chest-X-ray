# Experiments archive

This folder keeps **Research Paper (RP) experiment snapshots** organized without changing the repository root.

Original git branches were **archived here and not deleted**.

## Layout

```text
experiments/
  RP1/     # early architecture trainings (30 epochs / patience)
  RP2/     # augmentation, anti-overfitting, reporting, IEEE Access revision
```

## RP1

| Folder | Source branch |
|--------|----------------|
| [RP1/ResNet18-30epochs](RP1/ResNet18-30epochs/) | `Resnet18-30epochs-RP1` |
| [RP1/ResNet18-30epochs-patience](RP1/ResNet18-30epochs-patience/) | `Resnet18-30epochs-patience-RP1` |
| [RP1/DenseNet121-30epochs-patience](RP1/DenseNet121-30epochs-patience/) | `DenseNet121-30epochs-patience-RP1` |
| [RP1/VGG19-30epochs-patience](RP1/VGG19-30epochs-patience/) | `VGG19-30epochs-patience-RP1` |
| [RP1/MobileNetV3-30epochs-patience](RP1/MobileNetV3-30epochs-patience/) | `MobileNetV3-30epochs-patience-RP1` |
| [RP1/EfficientNetB0-30epochs-patience](RP1/EfficientNetB0-30epochs-patience/) | `EfficientNetB0-30epochs-patience-RP1` |

Each folder is a full snapshot of that branch tip (scripts, graphs, reports, weights when tracked). See `SOURCE_BRANCH.md` inside each folder.

## RP2

| Folder | Source |
|--------|--------|
| [RP2/augmentation](RP2/augmentation/) | branch `augmentation` |
| [RP2/ResNet18-anti-overfitting-ep17](RP2/ResNet18-anti-overfitting-ep17/) | branch `Resnet18-30epochs-17epochs-RP2` |
| [RP2/reporting](RP2/reporting/) | branch `RP2-reporting` (k-fold / external validation reports) |
| [RP2/ieee-access-revision](RP2/ieee-access-revision/) | IEEE Access revision code (`revision/`, reviewer docs); large checkpoints excluded |

## Notes

- Root app / README of `main` stay as the default project entrypoint.
- To reproduce an old experiment, `cd` into its folder and use that folder’s scripts/README.
