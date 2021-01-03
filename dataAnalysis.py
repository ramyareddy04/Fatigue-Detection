import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

txt = """2021-01-01 13:17:59.900131: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Epoch 1/50
136/136 [==============================] - 7s 54ms/step - loss: 0.6272 - accuracy: 0.6224 - val_loss: 0.5825 - val_accuracy: 0.8167
Epoch 2/50
136/136 [==============================] - 7s 52ms/step - loss: 0.4461 - accuracy: 0.8088 - val_loss: 0.2865 - val_accuracy: 0.9563
Epoch 3/50
136/136 [==============================] - 7s 52ms/step - loss: 0.3377 - accuracy: 0.8767 - val_loss: 0.2722 - val_accuracy: 0.8917
Epoch 4/50
136/136 [==============================] - 7s 52ms/step - loss: 0.2891 - accuracy: 0.8947 - val_loss: 0.1507 - val_accuracy: 0.9771
Epoch 5/50
136/136 [==============================] - 7s 52ms/step - loss: 0.2631 - accuracy: 0.9125 - val_loss: 0.3217 - val_accuracy: 0.9021
Epoch 6/50
136/136 [==============================] - 7s 52ms/step - loss: 0.2371 - accuracy: 0.9240 - val_loss: 0.2435 - val_accuracy: 0.9021
Epoch 7/50
136/136 [==============================] - 7s 52ms/step - loss: 0.2230 - accuracy: 0.9277 - val_loss: 0.1405 - val_accuracy: 0.9458
Epoch 8/50
136/136 [==============================] - 7s 53ms/step - loss: 0.2150 - accuracy: 0.9339 - val_loss: 0.1638 - val_accuracy: 0.9417
Epoch 9/50
136/136 [==============================] - 7s 53ms/step - loss: 0.1976 - accuracy: 0.9363 - val_loss: 0.1350 - val_accuracy: 0.9583
Epoch 10/50
136/136 [==============================] - 7s 53ms/step - loss: 0.1839 - accuracy: 0.9439 - val_loss: 0.1251 - val_accuracy: 0.9604
Epoch 11/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1883 - accuracy: 0.9418 - val_loss: 0.1062 - val_accuracy: 0.9708
Epoch 12/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1690 - accuracy: 0.9494 - val_loss: 0.0963 - val_accuracy: 0.9688
Epoch 13/50
136/136 [==============================] - 8s 60ms/step - loss: 0.1723 - accuracy: 0.9450 - val_loss: 0.0844 - val_accuracy: 0.9792
Epoch 14/50
136/136 [==============================] - 7s 55ms/step - loss: 0.1541 - accuracy: 0.9540 - val_loss: 0.1653 - val_accuracy: 0.9417
Epoch 15/50
136/136 [==============================] - 8s 59ms/step - loss: 0.1527 - accuracy: 0.9536 - val_loss: 0.0918 - val_accuracy: 0.9750
Epoch 16/50
136/136 [==============================] - 8s 58ms/step - loss: 0.1600 - accuracy: 0.9483 - val_loss: 0.1643 - val_accuracy: 0.9396
Epoch 17/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1561 - accuracy: 0.9499 - val_loss: 0.1421 - val_accuracy: 0.9479
Epoch 18/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1474 - accuracy: 0.9522 - val_loss: 0.0976 - val_accuracy: 0.9729
Epoch 19/50
136/136 [==============================] - 8s 57ms/step - loss: 0.1318 - accuracy: 0.9545 - val_loss: 0.1260 - val_accuracy: 0.9667
Epoch 20/50
136/136 [==============================] - 7s 55ms/step - loss: 0.1283 - accuracy: 0.9610 - val_loss: 0.0487 - val_accuracy: 0.9875
Epoch 21/50
136/136 [==============================] - 8s 62ms/step - loss: 0.1343 - accuracy: 0.9589 - val_loss: 0.1377 - val_accuracy: 0.9625
Epoch 22/50
136/136 [==============================] - 8s 58ms/step - loss: 0.1297 - accuracy: 0.9566 - val_loss: 0.1066 - val_accuracy: 0.9688
Epoch 23/50
136/136 [==============================] - 8s 55ms/step - loss: 0.1281 - accuracy: 0.9589 - val_loss: 0.1032 - val_accuracy: 0.9688
Epoch 24/50
136/136 [==============================] - 8s 57ms/step - loss: 0.1204 - accuracy: 0.9587 - val_loss: 0.0725 - val_accuracy: 0.9750
Epoch 25/50
136/136 [==============================] - 8s 59ms/step - loss: 0.1169 - accuracy: 0.9614 - val_loss: 0.0461 - val_accuracy: 0.9896
Epoch 26/50
136/136 [==============================] - 7s 55ms/step - loss: 0.1162 - accuracy: 0.9605 - val_loss: 0.1699 - val_accuracy: 0.9500
Epoch 27/50
136/136 [==============================] - 8s 62ms/step - loss: 0.1169 - accuracy: 0.9584 - val_loss: 0.1122 - val_accuracy: 0.9667
Epoch 28/50
136/136 [==============================] - 8s 59ms/step - loss: 0.1183 - accuracy: 0.9591 - val_loss: 0.0447 - val_accuracy: 0.9917
Epoch 29/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1068 - accuracy: 0.9658 - val_loss: 0.0933 - val_accuracy: 0.9750
Epoch 30/50
136/136 [==============================] - 8s 58ms/step - loss: 0.0975 - accuracy: 0.9707 - val_loss: 0.1264 - val_accuracy: 0.9604
Epoch 31/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1052 - accuracy: 0.9661 - val_loss: 0.2845 - val_accuracy: 0.9292
Epoch 32/50
136/136 [==============================] - 7s 54ms/step - loss: 0.1015 - accuracy: 0.9656 - val_loss: 0.1386 - val_accuracy: 0.9563
Epoch 33/50
136/136 [==============================] - 8s 59ms/step - loss: 0.0955 - accuracy: 0.9658 - val_loss: 0.0944 - val_accuracy: 0.9688
Epoch 34/50
136/136 [==============================] - 8s 56ms/step - loss: 0.1025 - accuracy: 0.9647 - val_loss: 0.1227 - val_accuracy: 0.9604
Epoch 35/50
136/136 [==============================] - 8s 58ms/step - loss: 0.1015 - accuracy: 0.9655 - val_loss: 0.1210 - val_accuracy: 0.9521
Epoch 36/50
136/136 [==============================] - 10s 70ms/step - loss: 0.1032 - accuracy: 0.9661 - val_loss: 0.0805 - val_accuracy: 0.9792
Epoch 37/50
136/136 [==============================] - 10s 74ms/step - loss: 0.1043 - accuracy: 0.9626 - val_loss: 0.1991 - val_accuracy: 0.9375
Epoch 38/50
136/136 [==============================] - 10s 72ms/step - loss: 0.0876 - accuracy: 0.9725 - val_loss: 0.1198 - val_accuracy: 0.9667
Epoch 39/50
136/136 [==============================] - 9s 69ms/step - loss: 0.0890 - accuracy: 0.9707 - val_loss: 0.1164 - val_accuracy: 0.9625
Epoch 40/50
136/136 [==============================] - 9s 66ms/step - loss: 0.0788 - accuracy: 0.9748 - val_loss: 0.1465 - val_accuracy: 0.9688
Epoch 41/50
136/136 [==============================] - 9s 65ms/step - loss: 0.0902 - accuracy: 0.9667 - val_loss: 0.1687 - val_accuracy: 0.9458
Epoch 42/50
136/136 [==============================] - 9s 69ms/step - loss: 0.0904 - accuracy: 0.9691 - val_loss: 0.1126 - val_accuracy: 0.9667
Epoch 43/50
136/136 [==============================] - 8s 55ms/step - loss: 0.0847 - accuracy: 0.9684 - val_loss: 0.1424 - val_accuracy: 0.9563
Epoch 44/50
136/136 [==============================] - 7s 54ms/step - loss: 0.0758 - accuracy: 0.9718 - val_loss: 0.2067 - val_accuracy: 0.9500
Epoch 45/50
136/136 [==============================] - 8s 61ms/step - loss: 0.0852 - accuracy: 0.9709 - val_loss: 0.0477 - val_accuracy: 0.9833
Epoch 46/50
136/136 [==============================] - 8s 57ms/step - loss: 0.0858 - accuracy: 0.9709 - val_loss: 0.0994 - val_accuracy: 0.9708
Epoch 47/50
136/136 [==============================] - 7s 53ms/step - loss: 0.0826 - accuracy: 0.9723 - val_loss: 0.0790 - val_accuracy: 0.9750
Epoch 48/50
136/136 [==============================] - 7s 53ms/step - loss: 0.0755 - accuracy: 0.9739 - val_loss: 0.0907 - val_accuracy: 0.9646
Epoch 49/50
136/136 [==============================] - 7s 52ms/step - loss: 0.0735 - accuracy: 0.9774 - val_loss: 0.1033 - val_accuracy: 0.9771
Epoch 50/50
136/136 [==============================] - 7s 53ms/step - loss: 0.0767 - accuracy: 0.9730 - val_loss: 0.0978 - val_accuracy: 0.9688"""

data = txt.split(" - ")

loss = []
accuracy = []
val_loss = []
val_accuracy = []
for i in data:
    if i.find("loss: ") != -1 and i.find("val_loss: ") != 0:
        loss.append(float(i[6:]))
    elif i.find("val_loss: ") == 0:
        val_loss.append(float(i[10:]))
    elif i.find("accuracy: ") != -1 and i.find("val_accuracy: ") != 0:
        accuracy.append(float(i[10:]))
    elif i.find("val_accuracy: ") == 0:
        val_accuracy.append(float(i[14:20]))


print(len(loss), len(val_loss), len(accuracy), len(val_accuracy))
print(loss)
fig, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(loss, 'g', label='Training loss')
ax1.plot(val_loss, 'b', label='validation loss')
ax1.set_title('Training and Validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(accuracy, 'g', label='Training accuracy')
ax2.plot(val_accuracy, 'b', label='validation accuracy')
ax2.set_title('Training and Validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()
