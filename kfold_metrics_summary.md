# 5-Fold Cross-Validation: Detailed Per-Fold Metrics

This document lists the confusion matrix components (TN, FP, FN, TP) and Recall (Sensitivity) for each fold across all 10 architectures.

## ResNet-18

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 4024 | 298 | 521 | 1465 | 0.7377 |
| Fold 2 | 4062 | 260 | 604 | 1382 | 0.6959 |
| Fold 3 | 4020 | 302 | 598 | 1388 | 0.6989 |
| Fold 4 | 4064 | 257 | 585 | 1402 | 0.7056 |
| Fold 5 | 3995 | 326 | 541 | 1446 | 0.7277 |

## ResNet-50

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 3991 | 331 | 530 | 1456 | 0.7331 |
| Fold 2 | 4041 | 281 | 640 | 1346 | 0.6777 |
| Fold 3 | 4034 | 288 | 600 | 1386 | 0.6979 |
| Fold 4 | 4007 | 314 | 547 | 1440 | 0.7247 |
| Fold 5 | 4051 | 270 | 595 | 1392 | 0.7006 |

## DenseNet-121

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 4063 | 259 | 610 | 1376 | 0.6928 |
| Fold 2 | 4054 | 268 | 592 | 1394 | 0.7019 |
| Fold 3 | 4047 | 275 | 601 | 1385 | 0.6974 |
| Fold 4 | 4070 | 251 | 603 | 1384 | 0.6965 |
| Fold 5 | 4067 | 254 | 586 | 1401 | 0.7051 |

## VGG-19

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 3970 | 352 | 593 | 1393 | 0.7014 |
| Fold 2 | 3988 | 334 | 639 | 1347 | 0.6782 |
| Fold 3 | 4322 | 0 | 1986 | 0 | 0.0000 |
| Fold 4 | 4321 | 0 | 1987 | 0 | 0.0000 |
| Fold 5 | 4321 | 0 | 1987 | 0 | 0.0000 |

## EfficientNet-B0

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 4046 | 276 | 527 | 1459 | 0.7346 |
| Fold 2 | 4035 | 287 | 565 | 1421 | 0.7155 |
| Fold 3 | 4054 | 268 | 595 | 1391 | 0.7004 |
| Fold 4 | 4027 | 294 | 517 | 1470 | 0.7398 |
| Fold 5 | 4023 | 298 | 514 | 1473 | 0.7413 |

## MobileNetV3-Small

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 4038 | 284 | 549 | 1437 | 0.7236 |
| Fold 2 | 4116 | 206 | 650 | 1336 | 0.6727 |
| Fold 3 | 4042 | 280 | 606 | 1380 | 0.6949 |
| Fold 4 | 4089 | 232 | 620 | 1367 | 0.6880 |
| Fold 5 | 3985 | 336 | 507 | 1480 | 0.7448 |

## ConvNeXt-Tiny

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 3982 | 340 | 633 | 1353 | 0.6813 |
| Fold 2 | 3916 | 406 | 567 | 1419 | 0.7145 |
| Fold 3 | 3931 | 391 | 705 | 1281 | 0.6450 |
| Fold 4 | 3961 | 360 | 595 | 1392 | 0.7006 |
| Fold 5 | 4116 | 205 | 797 | 1190 | 0.5989 |

## DeiT-Tiny

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 4012 | 310 | 679 | 1307 | 0.6581 |
| Fold 2 | 3984 | 338 | 724 | 1262 | 0.6354 |
| Fold 3 | 3921 | 401 | 643 | 1343 | 0.6762 |
| Fold 4 | 4019 | 302 | 678 | 1309 | 0.6588 |
| Fold 5 | 3886 | 435 | 619 | 1368 | 0.6885 |

## Swin-Tiny

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 3957 | 365 | 692 | 1294 | 0.6516 |
| Fold 2 | 4049 | 273 | 913 | 1073 | 0.5403 |
| Fold 3 | 4322 | 0 | 1986 | 0 | 0.0000 |
| Fold 4 | 3992 | 329 | 781 | 1206 | 0.6069 |
| Fold 5 | 4321 | 0 | 1987 | 0 | 0.0000 |

## ViT-B/16

| Fold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 3965 | 357 | 721 | 1265 | 0.6370 |
| Fold 2 | 3942 | 380 | 800 | 1186 | 0.5972 |
| Fold 3 | 3919 | 403 | 752 | 1234 | 0.6213 |
| Fold 4 | 3753 | 568 | 685 | 1302 | 0.6553 |
| Fold 5 | 3943 | 378 | 849 | 1138 | 0.5727 |

