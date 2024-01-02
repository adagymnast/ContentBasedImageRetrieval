import matplotlib.pyplot as plt
import numpy as np

s = """
2022-05-04 10:26:49,100 INFO [Train model] start
2022-05-04 10:40:01,819 DEBUG   Epoch 1 - avg_train_loss: 2.2918  avg_val_loss: 2.2828 F1: 0.176809  Accuracy: 0.179167 time: 793s
2022-05-04 10:40:01,819 DEBUG   Epoch 1 - Save Best Accuracy: 0.179167 Model
2022-05-04 10:40:02,178 DEBUG   Epoch 1 - Save Best Loss: 2.2828 Model
2022-05-04 10:54:52,836 DEBUG   Epoch 2 - avg_train_loss: 2.2731  avg_val_loss: 2.2639 F1: 0.303943  Accuracy: 0.312500 time: 890s
2022-05-04 10:54:52,836 DEBUG   Epoch 2 - Save Best Accuracy: 0.312500 Model
2022-05-04 10:54:53,242 DEBUG   Epoch 2 - Save Best Loss: 2.2639 Model
2022-05-04 11:09:08,705 DEBUG   Epoch 3 - avg_train_loss: 2.2454  avg_val_loss: 2.2394 F1: 0.402697  Accuracy: 0.408333 time: 855s
2022-05-04 11:09:08,706 DEBUG   Epoch 3 - Save Best Accuracy: 0.408333 Model
2022-05-04 11:09:09,193 DEBUG   Epoch 3 - Save Best Loss: 2.2394 Model
2022-05-04 11:23:28,280 DEBUG   Epoch 4 - avg_train_loss: 2.2182  avg_val_loss: 2.2110 F1: 0.469197  Accuracy: 0.470833 time: 859s
2022-05-04 11:23:28,280 DEBUG   Epoch 4 - Save Best Accuracy: 0.470833 Model
2022-05-04 11:23:28,673 DEBUG   Epoch 4 - Save Best Loss: 2.2110 Model
2022-05-04 11:37:45,232 DEBUG   Epoch 5 - avg_train_loss: 2.1913  avg_val_loss: 2.1803 F1: 0.621472  Accuracy: 0.616667 time: 856s
2022-05-04 11:37:45,232 DEBUG   Epoch 5 - Save Best Accuracy: 0.616667 Model
2022-05-04 11:37:45,629 DEBUG   Epoch 5 - Save Best Loss: 2.1803 Model
2022-05-04 11:52:42,957 DEBUG   Epoch 6 - avg_train_loss: 2.1508  avg_val_loss: 2.1385 F1: 0.666452  Accuracy: 0.662500 time: 897s
2022-05-04 11:52:42,957 DEBUG   Epoch 6 - Save Best Accuracy: 0.662500 Model
2022-05-04 11:52:43,371 DEBUG   Epoch 6 - Save Best Loss: 2.1385 Model
2022-05-04 12:07:05,959 DEBUG   Epoch 7 - avg_train_loss: 2.1066  avg_val_loss: 2.0976 F1: 0.692222  Accuracy: 0.695833 time: 862s
2022-05-04 12:07:05,959 DEBUG   Epoch 7 - Save Best Accuracy: 0.695833 Model
2022-05-04 12:07:06,364 DEBUG   Epoch 7 - Save Best Loss: 2.0976 Model
2022-05-04 12:18:49,751 DEBUG   Epoch 8 - avg_train_loss: 2.0635  avg_val_loss: 2.0470 F1: 0.760582  Accuracy: 0.758333 time: 703s
2022-05-04 12:18:49,751 DEBUG   Epoch 8 - Save Best Accuracy: 0.758333 Model
2022-05-04 12:18:50,097 DEBUG   Epoch 8 - Save Best Loss: 2.0470 Model
2022-05-04 12:30:38,536 DEBUG   Epoch 9 - avg_train_loss: 2.0109  avg_val_loss: 1.9848 F1: 0.784055  Accuracy: 0.783333 time: 708s
2022-05-04 12:30:38,536 DEBUG   Epoch 9 - Save Best Accuracy: 0.783333 Model
2022-05-04 12:30:38,941 DEBUG   Epoch 9 - Save Best Loss: 1.9848 Model
2022-05-04 12:44:48,884 DEBUG   Epoch 10 - avg_train_loss: 1.9508  avg_val_loss: 1.9192 F1: 0.796105  Accuracy: 0.795833 time: 850s
2022-05-04 12:44:48,884 DEBUG   Epoch 10 - Save Best Accuracy: 0.795833 Model
2022-05-04 12:44:49,306 DEBUG   Epoch 10 - Save Best Loss: 1.9192 Model
2022-05-04 12:58:49,748 DEBUG   Epoch 11 - avg_train_loss: 1.8759  avg_val_loss: 1.8428 F1: 0.806159  Accuracy: 0.804167 time: 840s
2022-05-04 12:58:49,748 DEBUG   Epoch 11 - Save Best Accuracy: 0.804167 Model
2022-05-04 12:58:50,030 DEBUG   Epoch 11 - Save Best Loss: 1.8428 Model
2022-05-04 13:12:52,905 DEBUG   Epoch 12 - avg_train_loss: 1.7805  avg_val_loss: 1.7476 F1: 0.837232  Accuracy: 0.837500 time: 843s
2022-05-04 13:12:52,907 DEBUG   Epoch 12 - Save Best Accuracy: 0.837500 Model
2022-05-04 13:12:53,281 DEBUG   Epoch 12 - Save Best Loss: 1.7476 Model
2022-05-04 13:24:20,571 DEBUG   Epoch 13 - avg_train_loss: 1.6887  avg_val_loss: 1.6414 F1: 0.846255  Accuracy: 0.845833 time: 687s
2022-05-04 13:24:20,571 DEBUG   Epoch 13 - Save Best Accuracy: 0.845833 Model
2022-05-04 13:24:20,948 DEBUG   Epoch 13 - Save Best Loss: 1.6414 Model
2022-05-04 13:37:34,881 DEBUG   Epoch 14 - avg_train_loss: 1.5901  avg_val_loss: 1.5121 F1: 0.857547  Accuracy: 0.858333 time: 794s
2022-05-04 13:37:34,881 DEBUG   Epoch 14 - Save Best Accuracy: 0.858333 Model
2022-05-04 13:37:35,468 DEBUG   Epoch 14 - Save Best Loss: 1.5121 Model
2022-05-04 13:51:38,194 DEBUG   Epoch 15 - avg_train_loss: 1.4804  avg_val_loss: 1.3822 F1: 0.871626  Accuracy: 0.870833 time: 842s
2022-05-04 13:51:38,194 DEBUG   Epoch 15 - Save Best Accuracy: 0.870833 Model
2022-05-04 13:51:38,603 DEBUG   Epoch 15 - Save Best Loss: 1.3822 Model
2022-05-04 14:05:51,887 DEBUG   Epoch 16 - avg_train_loss: 1.3665  avg_val_loss: 1.2503 F1: 0.876108  Accuracy: 0.875000 time: 853s
2022-05-04 14:05:51,887 DEBUG   Epoch 16 - Save Best Accuracy: 0.875000 Model
2022-05-04 14:05:52,246 DEBUG   Epoch 16 - Save Best Loss: 1.2503 Model
2022-05-04 14:21:00,223 DEBUG   Epoch 17 - avg_train_loss: 1.2512  avg_val_loss: 1.1437 F1: 0.883369  Accuracy: 0.883333 time: 908s
2022-05-04 14:21:00,223 DEBUG   Epoch 17 - Save Best Accuracy: 0.883333 Model
2022-05-04 14:21:00,636 DEBUG   Epoch 17 - Save Best Loss: 1.1437 Model
2022-05-04 14:36:34,295 DEBUG   Epoch 18 - avg_train_loss: 1.1427  avg_val_loss: 1.0142 F1: 0.888162  Accuracy: 0.887500 time: 933s
2022-05-04 14:36:34,295 DEBUG   Epoch 18 - Save Best Accuracy: 0.887500 Model
2022-05-04 14:36:34,689 DEBUG   Epoch 18 - Save Best Loss: 1.0142 Model
2022-05-04 14:51:08,699 DEBUG   Epoch 19 - avg_train_loss: 1.0472  avg_val_loss: 0.9199 F1: 0.880037  Accuracy: 0.879167 time: 874s
2022-05-04 14:51:08,699 DEBUG   Epoch 19 - Save Best Loss: 0.9199 Model
2022-05-04 15:05:47,346 DEBUG   Epoch 20 - avg_train_loss: 0.9524  avg_val_loss: 0.8368 F1: 0.909255  Accuracy: 0.908333 time: 878s
2022-05-04 15:05:47,346 DEBUG   Epoch 20 - Save Best Accuracy: 0.908333 Model
2022-05-04 15:05:47,795 DEBUG   Epoch 20 - Save Best Loss: 0.8368 Model
2022-05-04 15:21:34,567 DEBUG   Epoch 21 - avg_train_loss: 0.8837  avg_val_loss: 0.7470 F1: 0.908461  Accuracy: 0.908333 time: 946s
2022-05-04 15:21:34,568 DEBUG   Epoch 21 - Save Best Loss: 0.7470 Model
2022-05-04 15:36:38,844 DEBUG   Epoch 22 - avg_train_loss: 0.8166  avg_val_loss: 0.6816 F1: 0.912895  Accuracy: 0.912500 time: 904s
2022-05-04 15:36:38,845 DEBUG   Epoch 22 - Save Best Accuracy: 0.912500 Model
2022-05-04 15:36:39,256 DEBUG   Epoch 22 - Save Best Loss: 0.6816 Model
2022-05-04 15:51:02,641 DEBUG   Epoch 23 - avg_train_loss: 0.7532  avg_val_loss: 0.6296 F1: 0.920970  Accuracy: 0.920833 time: 863s
2022-05-04 15:51:02,641 DEBUG   Epoch 23 - Save Best Accuracy: 0.920833 Model
2022-05-04 15:51:03,145 DEBUG   Epoch 23 - Save Best Loss: 0.6296 Model
2022-05-04 16:06:09,370 DEBUG   Epoch 24 - avg_train_loss: 0.6988  avg_val_loss: 0.5864 F1: 0.921211  Accuracy: 0.920833 time: 906s
2022-05-04 16:06:09,370 DEBUG   Epoch 24 - Save Best Loss: 0.5864 Model
2022-05-04 16:20:28,071 DEBUG   Epoch 25 - avg_train_loss: 0.6374  avg_val_loss: 0.5190 F1: 0.937600  Accuracy: 0.937500 time: 858s
2022-05-04 16:20:28,087 DEBUG   Epoch 25 - Save Best Accuracy: 0.937500 Model
2022-05-04 16:20:28,472 DEBUG   Epoch 25 - Save Best Loss: 0.5190 Model
2022-05-04 16:35:26,345 DEBUG   Epoch 26 - avg_train_loss: 0.5979  avg_val_loss: 0.4840 F1: 0.937764  Accuracy: 0.937500 time: 898s
2022-05-04 16:35:26,345 DEBUG   Epoch 26 - Save Best Loss: 0.4840 Model
2022-05-04 16:48:27,667 DEBUG   Epoch 27 - avg_train_loss: 0.5645  avg_val_loss: 0.4471 F1: 0.945880  Accuracy: 0.945833 time: 781s
2022-05-04 16:48:27,667 DEBUG   Epoch 27 - Save Best Accuracy: 0.945833 Model
2022-05-04 16:48:28,058 DEBUG   Epoch 27 - Save Best Loss: 0.4471 Model
2022-05-04 17:01:59,075 DEBUG   Epoch 28 - avg_train_loss: 0.5114  avg_val_loss: 0.4369 F1: 0.949955  Accuracy: 0.950000 time: 811s
2022-05-04 17:01:59,075 DEBUG   Epoch 28 - Save Best Accuracy: 0.950000 Model
2022-05-04 17:01:59,375 DEBUG   Epoch 28 - Save Best Loss: 0.4369 Model
2022-05-04 17:15:26,686 DEBUG   Epoch 29 - avg_train_loss: 0.5201  avg_val_loss: 0.3869 F1: 0.946091  Accuracy: 0.945833 time: 807s
2022-05-04 17:15:26,687 DEBUG   Epoch 29 - Save Best Loss: 0.3869 Model
2022-05-04 17:29:39,092 DEBUG   Epoch 30 - avg_train_loss: 0.4652  avg_val_loss: 0.3566 F1: 0.945888  Accuracy: 0.945833 time: 852s
2022-05-04 17:29:39,092 DEBUG   Epoch 30 - Save Best Loss: 0.3566 Model
2022-05-04 17:42:51,578 DEBUG   Epoch 31 - avg_train_loss: 0.4621  avg_val_loss: 0.3704 F1: 0.933419  Accuracy: 0.933333 time: 792s
2022-05-04 17:56:35,968 DEBUG   Epoch 32 - avg_train_loss: 0.4212  avg_val_loss: 0.3271 F1: 0.950170  Accuracy: 0.950000 time: 824s
2022-05-04 17:56:35,969 DEBUG   Epoch 32 - Save Best Loss: 0.3271 Model
2022-05-04 18:09:57,468 DEBUG   Epoch 33 - avg_train_loss: 0.4636  avg_val_loss: 0.3115 F1: 0.941598  Accuracy: 0.941667 time: 801s
2022-05-04 18:09:57,468 DEBUG   Epoch 33 - Save Best Loss: 0.3115 Model
2022-05-04 18:24:02,837 DEBUG   Epoch 34 - avg_train_loss: 0.3963  avg_val_loss: 0.2871 F1: 0.958537  Accuracy: 0.958333 time: 845s
2022-05-04 18:24:02,837 DEBUG   Epoch 34 - Save Best Accuracy: 0.958333 Model
2022-05-04 18:24:03,134 DEBUG   Epoch 34 - Save Best Loss: 0.2871 Model
2022-05-04 18:36:09,019 DEBUG   Epoch 35 - avg_train_loss: 0.3972  avg_val_loss: 0.2900 F1: 0.946304  Accuracy: 0.945833 time: 726s
2022-05-04 18:47:34,914 DEBUG   Epoch 36 - avg_train_loss: 0.4033  avg_val_loss: 0.2823 F1: 0.946554  Accuracy: 0.945833 time: 686s
2022-05-04 18:47:34,914 DEBUG   Epoch 36 - Save Best Loss: 0.2823 Model
2022-05-04 18:59:03,401 DEBUG   Epoch 37 - avg_train_loss: 0.3365  avg_val_loss: 0.2641 F1: 0.954344  Accuracy: 0.954167 time: 688s
2022-05-04 18:59:03,401 DEBUG   Epoch 37 - Save Best Loss: 0.2641 Model
2022-05-04 19:10:29,952 DEBUG   Epoch 38 - avg_train_loss: 0.3422  avg_val_loss: 0.2417 F1: 0.958383  Accuracy: 0.958333 time: 686s
2022-05-04 19:10:29,952 DEBUG   Epoch 38 - Save Best Loss: 0.2417 Model
2022-05-04 19:21:57,176 DEBUG   Epoch 39 - avg_train_loss: 0.3343  avg_val_loss: 0.2359 F1: 0.962550  Accuracy: 0.962500 time: 687s
2022-05-04 19:21:57,176 DEBUG   Epoch 39 - Save Best Accuracy: 0.962500 Model
2022-05-04 19:21:57,504 DEBUG   Epoch 39 - Save Best Loss: 0.2359 Model
2022-05-04 19:33:25,496 DEBUG   Epoch 40 - avg_train_loss: 0.2977  avg_val_loss: 0.2275 F1: 0.954259  Accuracy: 0.954167 time: 687s
2022-05-04 19:33:25,496 DEBUG   Epoch 40 - Save Best Loss: 0.2275 Model
2022-05-04 19:44:49,851 DEBUG   Epoch 41 - avg_train_loss: 0.3457  avg_val_loss: 0.2070 F1: 0.962280  Accuracy: 0.962500 time: 684s
2022-05-04 19:44:49,851 DEBUG   Epoch 41 - Save Best Loss: 0.2070 Model
2022-05-04 19:56:14,254 DEBUG   Epoch 42 - avg_train_loss: 0.2454  avg_val_loss: 0.1987 F1: 0.966560  Accuracy: 0.966667 time: 684s
2022-05-04 19:56:14,254 DEBUG   Epoch 42 - Save Best Accuracy: 0.966667 Model
2022-05-04 19:56:14,598 DEBUG   Epoch 42 - Save Best Loss: 0.1987 Model
2022-05-04 20:07:35,914 DEBUG   Epoch 43 - avg_train_loss: 0.2568  avg_val_loss: 0.2221 F1: 0.954169  Accuracy: 0.954167 time: 681s
2022-05-04 20:18:58,749 DEBUG   Epoch 44 - avg_train_loss: 0.2630  avg_val_loss: 0.2081 F1: 0.954006  Accuracy: 0.954167 time: 683s
2022-05-04 20:31:25,015 DEBUG   Epoch 45 - avg_train_loss: 0.2801  avg_val_loss: 0.2053 F1: 0.954364  Accuracy: 0.954167 time: 746s
2022-05-04 20:45:09,139 DEBUG   Epoch 46 - avg_train_loss: 0.2602  avg_val_loss: 0.2018 F1: 0.954350  Accuracy: 0.954167 time: 824s
2022-05-04 20:59:55,570 DEBUG   Epoch 47 - avg_train_loss: 0.2321  avg_val_loss: 0.1843 F1: 0.962395  Accuracy: 0.962500 time: 886s
2022-05-04 20:59:55,570 DEBUG   Epoch 47 - Save Best Loss: 0.1843 Model
2022-05-04 21:16:02,193 DEBUG   Epoch 48 - avg_train_loss: 0.2560  avg_val_loss: 0.1978 F1: 0.962294  Accuracy: 0.962500 time: 966s
2022-05-04 21:33:47,115 DEBUG   Epoch 49 - avg_train_loss: 0.2600  avg_val_loss: 0.1730 F1: 0.966560  Accuracy: 0.966667 time: 1065s
2022-05-04 21:33:47,116 DEBUG   Epoch 49 - Save Best Loss: 0.1730 Model
2022-05-04 21:49:52,392 DEBUG   Epoch 50 - avg_train_loss: 0.2521  avg_val_loss: 0.1954 F1: 0.957946  Accuracy: 0.958333 time: 965s
2022-05-04 22:05:24,379 DEBUG   Epoch 51 - avg_train_loss: 0.1960  avg_val_loss: 0.1735 F1: 0.962489  Accuracy: 0.962500 time: 932s
2022-05-04 22:20:44,653 DEBUG   Epoch 52 - avg_train_loss: 0.2236  avg_val_loss: 0.1938 F1: 0.953783  Accuracy: 0.954167 time: 920s
2022-05-04 22:36:58,140 DEBUG   Epoch 53 - avg_train_loss: 0.2221  avg_val_loss: 0.1575 F1: 0.974893  Accuracy: 0.975000 time: 973s
2022-05-04 22:36:58,141 DEBUG   Epoch 53 - Save Best Accuracy: 0.975000 Model
2022-05-04 22:36:58,560 DEBUG   Epoch 53 - Save Best Loss: 0.1575 Model
2022-05-04 22:52:49,365 DEBUG   Epoch 54 - avg_train_loss: 0.2592  avg_val_loss: 0.1605 F1: 0.966628  Accuracy: 0.966667 time: 950s
2022-05-04 23:05:35,345 DEBUG   Epoch 55 - avg_train_loss: 0.2130  avg_val_loss: 0.1759 F1: 0.954235  Accuracy: 0.954167 time: 766s
2022-05-04 23:16:14,265 DEBUG   Epoch 56 - avg_train_loss: 0.2134  avg_val_loss: 0.1694 F1: 0.966560  Accuracy: 0.966667 time: 639s
2022-05-04 23:27:20,198 DEBUG   Epoch 57 - avg_train_loss: 0.2105  avg_val_loss: 0.1664 F1: 0.966567  Accuracy: 0.966667 time: 666s
2022-05-04 23:38:27,097 DEBUG   Epoch 58 - avg_train_loss: 0.2057  avg_val_loss: 0.1561 F1: 0.966569  Accuracy: 0.966667 time: 667s
2022-05-04 23:38:27,097 DEBUG   Epoch 58 - Save Best Loss: 0.1561 Model
2022-05-04 23:50:32,153 DEBUG   Epoch 59 - avg_train_loss: 0.1875  avg_val_loss: 0.1544 F1: 0.962569  Accuracy: 0.962500 time: 725s
2022-05-04 23:50:32,153 DEBUG   Epoch 59 - Save Best Loss: 0.1544 Model
2022-05-05 00:03:35,768 DEBUG   Epoch 60 - avg_train_loss: 0.1949  avg_val_loss: 0.1679 F1: 0.950142  Accuracy: 0.950000 time: 783s
2022-05-05 00:15:33,826 DEBUG   Epoch 61 - avg_train_loss: 0.2139  avg_val_loss: 0.1580 F1: 0.954722  Accuracy: 0.954167 time: 718s
2022-05-05 00:29:50,398 DEBUG   Epoch 62 - avg_train_loss: 0.2020  avg_val_loss: 0.1444 F1: 0.962637  Accuracy: 0.962500 time: 857s
2022-05-05 00:29:50,398 DEBUG   Epoch 62 - Save Best Loss: 0.1444 Model
2022-05-05 00:45:19,025 DEBUG   Epoch 63 - avg_train_loss: 0.1676  avg_val_loss: 0.1462 F1: 0.958381  Accuracy: 0.958333 time: 928s
2022-05-05 01:00:35,473 DEBUG   Epoch 64 - avg_train_loss: 0.1709  avg_val_loss: 0.1581 F1: 0.958312  Accuracy: 0.958333 time: 916s
2022-05-05 01:13:33,851 DEBUG   Epoch 65 - avg_train_loss: 0.2213  avg_val_loss: 0.1479 F1: 0.954220  Accuracy: 0.954167 time: 778s
2022-05-05 01:27:54,313 DEBUG   Epoch 66 - avg_train_loss: 0.1724  avg_val_loss: 0.1548 F1: 0.962388  Accuracy: 0.962500 time: 860s
2022-05-05 01:42:17,417 DEBUG   Epoch 67 - avg_train_loss: 0.1726  avg_val_loss: 0.1379 F1: 0.966560  Accuracy: 0.966667 time: 863s
2022-05-05 01:42:17,417 DEBUG   Epoch 67 - Save Best Loss: 0.1379 Model
2022-05-05 01:56:29,923 DEBUG   Epoch 68 - avg_train_loss: 0.1485  avg_val_loss: 0.1391 F1: 0.975052  Accuracy: 0.975000 time: 852s
2022-05-05 02:10:48,037 DEBUG   Epoch 69 - avg_train_loss: 0.1567  avg_val_loss: 0.1582 F1: 0.958557  Accuracy: 0.958333 time: 858s
2022-05-05 02:24:04,230 DEBUG   Epoch 70 - avg_train_loss: 0.1245  avg_val_loss: 0.1421 F1: 0.962484  Accuracy: 0.962500 time: 796s
2022-05-05 02:38:27,081 DEBUG   Epoch 71 - avg_train_loss: 0.1826  avg_val_loss: 0.1395 F1: 0.970810  Accuracy: 0.970833 time: 863s
2022-05-05 02:50:35,309 DEBUG   Epoch 72 - avg_train_loss: 0.1583  avg_val_loss: 0.1468 F1: 0.958405  Accuracy: 0.958333 time: 728s
2022-05-05 03:03:25,043 DEBUG   Epoch 73 - avg_train_loss: 0.1876  avg_val_loss: 0.1326 F1: 0.966654  Accuracy: 0.966667 time: 770s
2022-05-05 03:03:25,044 DEBUG   Epoch 73 - Save Best Loss: 0.1326 Model
2022-05-05 03:16:32,567 DEBUG   Epoch 74 - avg_train_loss: 0.1360  avg_val_loss: 0.1253 F1: 0.970913  Accuracy: 0.970833 time: 787s
2022-05-05 03:16:32,568 DEBUG   Epoch 74 - Save Best Loss: 0.1253 Model
2022-05-05 03:31:13,980 DEBUG   Epoch 75 - avg_train_loss: 0.1334  avg_val_loss: 0.1297 F1: 0.962397  Accuracy: 0.962500 time: 881s
2022-05-05 03:44:52,312 DEBUG   Epoch 76 - avg_train_loss: 0.1383  avg_val_loss: 0.1298 F1: 0.966652  Accuracy: 0.966667 time: 818s
2022-05-05 03:59:23,790 DEBUG   Epoch 77 - avg_train_loss: 0.1553  avg_val_loss: 0.1357 F1: 0.962314  Accuracy: 0.962500 time: 871s
2022-05-05 04:13:39,564 DEBUG   Epoch 78 - avg_train_loss: 0.1293  avg_val_loss: 0.1552 F1: 0.954015  Accuracy: 0.954167 time: 856s
2022-05-05 04:27:50,929 DEBUG   Epoch 79 - avg_train_loss: 0.1577  avg_val_loss: 0.1413 F1: 0.957951  Accuracy: 0.958333 time: 851s
2022-05-05 04:41:13,254 DEBUG   Epoch 80 - avg_train_loss: 0.1497  avg_val_loss: 0.1142 F1: 0.970904  Accuracy: 0.970833 time: 802s
2022-05-05 04:41:13,254 DEBUG   Epoch 80 - Save Best Loss: 0.1142 Model
2022-05-05 04:55:35,657 DEBUG   Epoch 81 - avg_train_loss: 0.1362  avg_val_loss: 0.1361 F1: 0.966553  Accuracy: 0.966667 time: 862s
2022-05-05 05:10:21,923 DEBUG   Epoch 82 - avg_train_loss: 0.1385  avg_val_loss: 0.1369 F1: 0.966652  Accuracy: 0.966667 time: 886s
2022-05-05 05:23:46,760 DEBUG   Epoch 83 - avg_train_loss: 0.1146  avg_val_loss: 0.1451 F1: 0.962004  Accuracy: 0.962500 time: 805s
2022-05-05 05:38:36,911 DEBUG   Epoch 84 - avg_train_loss: 0.1381  avg_val_loss: 0.1333 F1: 0.970734  Accuracy: 0.970833 time: 890s
2022-05-05 05:53:08,872 DEBUG   Epoch 85 - avg_train_loss: 0.1154  avg_val_loss: 0.1213 F1: 0.966563  Accuracy: 0.966667 time: 872s
2022-05-05 06:08:01,573 DEBUG   Epoch 86 - avg_train_loss: 0.1097  avg_val_loss: 0.1186 F1: 0.966645  Accuracy: 0.966667 time: 893s
2022-05-05 06:23:46,253 DEBUG   Epoch 87 - avg_train_loss: 0.1152  avg_val_loss: 0.1308 F1: 0.958296  Accuracy: 0.958333 time: 945s
2022-05-05 06:39:56,160 DEBUG   Epoch 88 - avg_train_loss: 0.1173  avg_val_loss: 0.1496 F1: 0.962393  Accuracy: 0.962500 time: 970s
2022-05-05 06:55:12,874 DEBUG   Epoch 89 - avg_train_loss: 0.1018  avg_val_loss: 0.1355 F1: 0.962408  Accuracy: 0.962500 time: 917s
2022-05-05 07:09:46,806 DEBUG   Epoch 90 - avg_train_loss: 0.1447  avg_val_loss: 0.1455 F1: 0.958064  Accuracy: 0.958333 time: 874s
2022-05-05 07:24:48,579 DEBUG   Epoch 91 - avg_train_loss: 0.1404  avg_val_loss: 0.1245 F1: 0.954319  Accuracy: 0.954167 time: 902s
2022-05-05 07:39:01,478 DEBUG   Epoch 92 - avg_train_loss: 0.1291  avg_val_loss: 0.1352 F1: 0.958482  Accuracy: 0.958333 time: 853s
2022-05-05 07:51:31,218 DEBUG   Epoch 93 - avg_train_loss: 0.1463  avg_val_loss: 0.1329 F1: 0.954317  Accuracy: 0.954167 time: 750s
2022-05-05 08:05:07,942 DEBUG   Epoch 94 - avg_train_loss: 0.0888  avg_val_loss: 0.1446 F1: 0.954395  Accuracy: 0.954167 time: 817s
2022-05-05 08:18:16,917 DEBUG   Epoch 95 - avg_train_loss: 0.0873  avg_val_loss: 0.1392 F1: 0.966384  Accuracy: 0.966667 time: 789s
2022-05-05 08:32:11,981 DEBUG   Epoch 96 - avg_train_loss: 0.1075  avg_val_loss: 0.1425 F1: 0.958402  Accuracy: 0.958333 time: 835s
2022-05-05 08:46:12,883 DEBUG   Epoch 97 - avg_train_loss: 0.0987  avg_val_loss: 0.1197 F1: 0.966654  Accuracy: 0.966667 time: 841s
2022-05-05 08:58:45,283 DEBUG   Epoch 98 - avg_train_loss: 0.0912  avg_val_loss: 0.1227 F1: 0.966820  Accuracy: 0.966667 time: 752s
2022-05-05 09:12:32,253 DEBUG   Epoch 99 - avg_train_loss: 0.0838  avg_val_loss: 0.1228 F1: 0.962402  Accuracy: 0.962500 time: 827s
2022-05-05 09:27:28,282 DEBUG   Epoch 100 - avg_train_loss: 0.1083  avg_val_loss: 0.1001 F1: 0.979069  Accuracy: 0.979167 time: 896s
2022-05-05 09:27:28,283 DEBUG   Epoch 100 - Save Best Accuracy: 0.979167 Model
2022-05-05 09:27:28,694 DEBUG   Epoch 100 - Save Best Loss: 0.1001 Model
2022-05-05 09:42:03,184 DEBUG   Epoch 101 - avg_train_loss: 0.0809  avg_val_loss: 0.1084 F1: 0.970726  Accuracy: 0.970833 time: 874s
2022-05-05 09:56:52,200 DEBUG   Epoch 102 - avg_train_loss: 0.0753  avg_val_loss: 0.1384 F1: 0.958482  Accuracy: 0.958333 time: 889s
2022-05-05 10:08:39,340 DEBUG   Epoch 103 - avg_train_loss: 0.0942  avg_val_loss: 0.1415 F1: 0.958324  Accuracy: 0.958333 time: 707s
2022-05-05 10:24:15,614 DEBUG   Epoch 104 - avg_train_loss: 0.0717  avg_val_loss: 0.1248 F1: 0.962480  Accuracy: 0.962500 time: 936s
2022-05-05 10:38:23,640 DEBUG   Epoch 105 - avg_train_loss: 0.1073  avg_val_loss: 0.1242 F1: 0.962484  Accuracy: 0.962500 time: 848s
2022-05-05 10:52:40,491 DEBUG   Epoch 106 - avg_train_loss: 0.0871  avg_val_loss: 0.1142 F1: 0.974813  Accuracy: 0.975000 time: 857s
2022-05-05 11:07:10,558 DEBUG   Epoch 107 - avg_train_loss: 0.0653  avg_val_loss: 0.1205 F1: 0.962127  Accuracy: 0.962500 time: 870s
2022-05-05 11:22:27,409 DEBUG   Epoch 108 - avg_train_loss: 0.0941  avg_val_loss: 0.1182 F1: 0.966654  Accuracy: 0.966667 time: 917s
2022-05-05 11:37:12,740 DEBUG   Epoch 109 - avg_train_loss: 0.0984  avg_val_loss: 0.1119 F1: 0.970562  Accuracy: 0.970833 time: 885s
2022-05-05 11:50:10,844 DEBUG   Epoch 110 - avg_train_loss: 0.1048  avg_val_loss: 0.1072 F1: 0.962393  Accuracy: 0.962500 time: 778s
2022-05-05 12:04:25,973 DEBUG   Epoch 111 - avg_train_loss: 0.0725  avg_val_loss: 0.1083 F1: 0.970902  Accuracy: 0.970833 time: 855s
2022-05-05 12:21:08,619 DEBUG   Epoch 112 - avg_train_loss: 0.0867  avg_val_loss: 0.1094 F1: 0.966381  Accuracy: 0.966667 time: 1003s
2022-05-05 12:36:06,028 DEBUG   Epoch 113 - avg_train_loss: 0.0659  avg_val_loss: 0.1067 F1: 0.966654  Accuracy: 0.966667 time: 897s
2022-05-05 12:49:47,366 DEBUG   Epoch 114 - avg_train_loss: 0.0581  avg_val_loss: 0.1204 F1: 0.974893  Accuracy: 0.975000 time: 821s
2022-05-05 13:02:47,875 DEBUG   Epoch 115 - avg_train_loss: 0.0699  avg_val_loss: 0.1092 F1: 0.970904  Accuracy: 0.970833 time: 781s
2022-05-05 13:16:45,806 DEBUG   Epoch 116 - avg_train_loss: 0.0796  avg_val_loss: 0.1213 F1: 0.966480  Accuracy: 0.966667 time: 838s
2022-05-05 13:31:18,602 DEBUG   Epoch 117 - avg_train_loss: 0.0857  avg_val_loss: 0.0990 F1: 0.974984  Accuracy: 0.975000 time: 873s
2022-05-05 13:31:18,602 DEBUG   Epoch 117 - Save Best Loss: 0.0990 Model
2022-05-05 13:45:57,424 DEBUG   Epoch 118 - avg_train_loss: 0.0770  avg_val_loss: 0.1296 F1: 0.962223  Accuracy: 0.962500 time: 878s
2022-05-05 14:01:40,719 DEBUG   Epoch 119 - avg_train_loss: 0.0745  avg_val_loss: 0.1117 F1: 0.962397  Accuracy: 0.962500 time: 943s
2022-05-05 14:17:49,035 DEBUG   Epoch 120 - avg_train_loss: 0.0788  avg_val_loss: 0.1370 F1: 0.958321  Accuracy: 0.958333 time: 968s
2022-05-05 14:34:18,557 DEBUG   Epoch 121 - avg_train_loss: 0.0741  avg_val_loss: 0.1507 F1: 0.958547  Accuracy: 0.958333 time: 990s
2022-05-05 14:52:21,390 DEBUG   Epoch 122 - avg_train_loss: 0.0551  avg_val_loss: 0.1147 F1: 0.970820  Accuracy: 0.970833 time: 1083s
2022-05-05 15:09:42,751 DEBUG   Epoch 123 - avg_train_loss: 0.0660  avg_val_loss: 0.1143 F1: 0.979059  Accuracy: 0.979167 time: 1041s
2022-05-05 15:24:37,903 DEBUG   Epoch 124 - avg_train_loss: 0.0531  avg_val_loss: 0.1490 F1: 0.953964  Accuracy: 0.954167 time: 895s
2022-05-05 15:38:03,609 DEBUG   Epoch 125 - avg_train_loss: 0.0669  avg_val_loss: 0.1246 F1: 0.958471  Accuracy: 0.958333 time: 806s
2022-05-05 15:51:06,736 DEBUG   Epoch 126 - avg_train_loss: 0.0735  avg_val_loss: 0.1138 F1: 0.966649  Accuracy: 0.966667 time: 783s
2022-05-05 16:03:01,706 DEBUG   Epoch 127 - avg_train_loss: 0.0751  avg_val_loss: 0.1196 F1: 0.966904  Accuracy: 0.966667 time: 715s
2022-05-05 16:15:44,066 DEBUG   Epoch 128 - avg_train_loss: 0.0581  avg_val_loss: 0.1055 F1: 0.962404  Accuracy: 0.962500 time: 762s
2022-05-05 16:29:37,681 DEBUG   Epoch 129 - avg_train_loss: 0.0687  avg_val_loss: 0.1237 F1: 0.966654  Accuracy: 0.966667 time: 834s
2022-05-05 16:50:00,699 DEBUG   Epoch 130 - avg_train_loss: 0.0627  avg_val_loss: 0.1090 F1: 0.970652  Accuracy: 0.970833 time: 1223s
2022-05-05 17:07:35,301 DEBUG   Epoch 131 - avg_train_loss: 0.0468  avg_val_loss: 0.1290 F1: 0.954074  Accuracy: 0.954167 time: 1055s
"""

arr = s.split(' ')
epochs = []
train_losses = []
val_losses = []
accuracies = []

for i, element in enumerate(arr):
    if element == 'Epoch':
        epochs.append(arr[i+1])
    if element == 'avg_val_loss:':
        val_losses.append(arr[i+1])
    if element == 'avg_train_loss:':
        train_losses.append(arr[i+1])
    if element == 'Accuracy:' and arr[i-1] != 'Best':
        accuracies.append(arr[i+1])


train_losses = np.array(train_losses)
train_losses = train_losses.astype(np.float64)

val_losses = np.array(val_losses)
val_losses = val_losses.astype(np.float64)

accuracies = np.array(accuracies)
accuracies = accuracies.astype(np.float64)

epochs = range(len(train_losses))
plt.plot(epochs, train_losses, 'g', label='Training loss')
plt.plot(epochs, val_losses, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

epochs = range(len(accuracies))
plt.plot(epochs, accuracies, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
