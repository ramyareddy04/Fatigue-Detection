import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

txt = """Epoch 1/100
121/121 [==============================] - 20s 168ms/step - loss: 0.5131 - accuracy: 0.7337 - val_loss: 1.0478 - val_accuracy: 0.5344
Epoch 2/100
121/121 [==============================] - 12s 102ms/step - loss: 0.2886 - accuracy: 0.8965 - val_loss: 0.4391 - val_accuracy: 0.8667
Epoch 3/100
121/121 [==============================] - 12s 103ms/step - loss: 0.1996 - accuracy: 0.9287 - val_loss: 0.3149 - val_accuracy: 0.8927
Epoch 4/100
121/121 [==============================] - 13s 108ms/step - loss: 0.1628 - accuracy: 0.9410 - val_loss: 0.2881 - val_accuracy: 0.9000
Epoch 5/100
121/121 [==============================] - 12s 98ms/step - loss: 0.1321 - accuracy: 0.9516 - val_loss: 0.1860 - val_accuracy: 0.9344
Epoch 6/100
121/121 [==============================] - 12s 96ms/step - loss: 0.1064 - accuracy: 0.9597 - val_loss: 0.3464 - val_accuracy: 0.8875
Epoch 7/100
121/121 [==============================] - 12s 96ms/step - loss: 0.1048 - accuracy: 0.9607 - val_loss: 0.2371 - val_accuracy: 0.9208
Epoch 8/100
121/121 [==============================] - 11s 91ms/step - loss: 0.0910 - accuracy: 0.9667 - val_loss: 0.3669 - val_accuracy: 0.8781
Epoch 9/100
121/121 [==============================] - 11s 94ms/step - loss: 0.0803 - accuracy: 0.9675 - val_loss: 0.5607 - val_accuracy: 0.8500
Epoch 10/100
121/121 [==============================] - 11s 93ms/step - loss: 0.0790 - accuracy: 0.9696 - val_loss: 0.2441 - val_accuracy: 0.9240
Epoch 11/100
121/121 [==============================] - 12s 96ms/step - loss: 0.0658 - accuracy: 0.9727 - val_loss: 0.3681 - val_accuracy: 0.8760
Epoch 12/100
121/121 [==============================] - 11s 93ms/step - loss: 0.0604 - accuracy: 0.9774 - val_loss: 0.3002 - val_accuracy: 0.9146
Epoch 13/100
121/121 [==============================] - 11s 91ms/step - loss: 0.0531 - accuracy: 0.9779 - val_loss: 0.2877 - val_accuracy: 0.9167
Epoch 14/100
121/121 [==============================] - 11s 94ms/step - loss: 0.0561 - accuracy: 0.9802 - val_loss: 0.2533 - val_accuracy: 0.9365
Epoch 15/100
121/121 [==============================] - 12s 96ms/step - loss: 0.0425 - accuracy: 0.9836 - val_loss: 0.3893 - val_accuracy: 0.9156
Epoch 16/100
121/121 [==============================] - 12s 100ms/step - loss: 0.0404 - accuracy: 0.9862 - val_loss: 0.2527 - val_accuracy: 0.9458
Epoch 17/100
121/121 [==============================] - 13s 104ms/step - loss: 0.0361 - accuracy: 0.9870 - val_loss: 0.4709 - val_accuracy: 0.8958
Epoch 18/100
121/121 [==============================] - 13s 107ms/step - loss: 0.0512 - accuracy: 0.9805 - val_loss: 0.2276 - val_accuracy: 0.9396
Epoch 19/100
121/121 [==============================] - 13s 104ms/step - loss: 0.0730 - accuracy: 0.9709 - val_loss: 0.3262 - val_accuracy: 0.9271
Epoch 20/100
121/121 [==============================] - 12s 98ms/step - loss: 0.0298 - accuracy: 0.9883 - val_loss: 0.5802 - val_accuracy: 0.8938
Epoch 21/100
121/121 [==============================] - 12s 100ms/step - loss: 0.0334 - accuracy: 0.9854 - val_loss: 0.5057 - val_accuracy: 0.8927
Epoch 22/100
121/121 [==============================] - 12s 99ms/step - loss: 0.0266 - accuracy: 0.9886 - val_loss: 0.3847 - val_accuracy: 0.9406
Epoch 23/100
121/121 [==============================] - 14s 115ms/step - loss: 0.0349 - accuracy: 0.9883 - val_loss: 0.2549 - val_accuracy: 0.9479
Epoch 24/100
121/121 [==============================] - 12s 101ms/step - loss: 0.0222 - accuracy: 0.9927 - val_loss: 0.4851 - val_accuracy: 0.9323
Epoch 25/100
121/121 [==============================] - 11s 91ms/step - loss: 0.0458 - accuracy: 0.9826 - val_loss: 0.4505 - val_accuracy: 0.9125
Epoch 26/100
121/121 [==============================] - 12s 96ms/step - loss: 0.0247 - accuracy: 0.9912 - val_loss: 0.5623 - val_accuracy: 0.9146
Epoch 27/100
121/121 [==============================] - 13s 107ms/step - loss: 0.0156 - accuracy: 0.9940 - val_loss: 0.6295 - val_accuracy: 0.9167
Epoch 28/100
121/121 [==============================] - 14s 112ms/step - loss: 0.0184 - accuracy: 0.9919 - val_loss: 0.5582 - val_accuracy: 0.9208
Epoch 29/100
121/121 [==============================] - 13s 111ms/step - loss: 0.0301 - accuracy: 0.9873 - val_loss: 0.5632 - val_accuracy: 0.9094
Epoch 30/100
121/121 [==============================] - 12s 102ms/step - loss: 0.0193 - accuracy: 0.9935 - val_loss: 0.3653 - val_accuracy: 0.9312
Epoch 31/100
121/121 [==============================] - 13s 108ms/step - loss: 0.0246 - accuracy: 0.9906 - val_loss: 0.4242 - val_accuracy: 0.9323
Epoch 32/100
121/121 [==============================] - 13s 103ms/step - loss: 0.0221 - accuracy: 0.9914 - val_loss: 0.4665 - val_accuracy: 0.9240
Epoch 33/100
121/121 [==============================] - 13s 104ms/step - loss: 0.0267 - accuracy: 0.9893 - val_loss: 0.4712 - val_accuracy: 0.9260
Epoch 34/100
121/121 [==============================] - 12s 97ms/step - loss: 0.0274 - accuracy: 0.9914 - val_loss: 0.5305 - val_accuracy: 0.9052
Epoch 35/100
121/121 [==============================] - 12s 103ms/step - loss: 0.0233 - accuracy: 0.9917 - val_loss: 0.5428 - val_accuracy: 0.9156
Epoch 36/100
121/121 [==============================] - 12s 99ms/step - loss: 0.0202 - accuracy: 0.9922 - val_loss: 0.8546 - val_accuracy: 0.8990
Epoch 37/100
121/121 [==============================] - 12s 98ms/step - loss: 0.0209 - accuracy: 0.9925 - val_loss: 0.7643 - val_accuracy: 0.9000
Epoch 38/100
121/121 [==============================] - 13s 111ms/step - loss: 0.0227 - accuracy: 0.9901 - val_loss: 0.6875 - val_accuracy: 0.9083
Epoch 39/100
121/121 [==============================] - 15s 123ms/step - loss: 0.0259 - accuracy: 0.9932 - val_loss: 0.3774 - val_accuracy: 0.9510
Epoch 40/100
121/121 [==============================] - 15s 127ms/step - loss: 0.0155 - accuracy: 0.9935 - val_loss: 0.8390 - val_accuracy: 0.9052
Epoch 41/100
121/121 [==============================] - 14s 112ms/step - loss: 0.0267 - accuracy: 0.9904 - val_loss: 0.5851 - val_accuracy: 0.9208
Epoch 42/100
121/121 [==============================] - 16s 130ms/step - loss: 0.0122 - accuracy: 0.9935 - val_loss: 0.6729 - val_accuracy: 0.9229
Epoch 43/100
121/121 [==============================] - 13s 104ms/step - loss: 0.0144 - accuracy: 0.9932 - val_loss: 0.6696 - val_accuracy: 0.9167
Epoch 44/100
121/121 [==============================] - 12s 97ms/step - loss: 0.0225 - accuracy: 0.9912 - val_loss: 0.9954 - val_accuracy: 0.8906
Epoch 45/100
121/121 [==============================] - 12s 99ms/step - loss: 0.0154 - accuracy: 0.9943 - val_loss: 0.9205 - val_accuracy: 0.9094
Epoch 46/100
121/121 [==============================] - 13s 105ms/step - loss: 0.0184 - accuracy: 0.9925 - val_loss: 0.7765 - val_accuracy: 0.9115
Epoch 47/100
121/121 [==============================] - 14s 113ms/step - loss: 0.0210 - accuracy: 0.9925 - val_loss: 0.5257 - val_accuracy: 0.9240
Epoch 48/100
121/121 [==============================] - 12s 101ms/step - loss: 0.0214 - accuracy: 0.9917 - val_loss: 1.0919 - val_accuracy: 0.8771
Epoch 49/100
121/121 [==============================] - 15s 124ms/step - loss: 0.0107 - accuracy: 0.9953 - val_loss: 0.8402 - val_accuracy: 0.9344
Epoch 50/100
121/121 [==============================] - 18s 153ms/step - loss: 0.0234 - accuracy: 0.9904 - val_loss: 0.5111 - val_accuracy: 0.9531
Epoch 51/100
121/121 [==============================] - 18s 145ms/step - loss: 0.0268 - accuracy: 0.9909 - val_loss: 0.7752 - val_accuracy: 0.8990
Epoch 52/100
121/121 [==============================] - 15s 125ms/step - loss: 0.0112 - accuracy: 0.9945 - val_loss: 0.6997 - val_accuracy: 0.9260
Epoch 53/100
121/121 [==============================] - 15s 127ms/step - loss: 0.0193 - accuracy: 0.9919 - val_loss: 0.8129 - val_accuracy: 0.9031
Epoch 54/100
121/121 [==============================] - 16s 131ms/step - loss: 0.0170 - accuracy: 0.9938 - val_loss: 0.6194 - val_accuracy: 0.9281
Epoch 55/100
121/121 [==============================] - 16s 130ms/step - loss: 0.0104 - accuracy: 0.9969 - val_loss: 0.7551 - val_accuracy: 0.9167
Epoch 56/100
121/121 [==============================] - 14s 116ms/step - loss: 0.0057 - accuracy: 0.9979 - val_loss: 0.7846 - val_accuracy: 0.9240
Epoch 57/100
121/121 [==============================] - 14s 115ms/step - loss: 0.0160 - accuracy: 0.9951 - val_loss: 1.1130 - val_accuracy: 0.8906
Epoch 58/100
121/121 [==============================] - 13s 106ms/step - loss: 0.0238 - accuracy: 0.9917 - val_loss: 0.4247 - val_accuracy: 0.9365
Epoch 59/100
121/121 [==============================] - 14s 116ms/step - loss: 0.0221 - accuracy: 0.9930 - val_loss: 1.0231 - val_accuracy: 0.8917
Epoch 60/100
121/121 [==============================] - 13s 108ms/step - loss: 0.0131 - accuracy: 0.9958 - val_loss: 0.9785 - val_accuracy: 0.9073
Epoch 61/100
121/121 [==============================] - 14s 117ms/step - loss: 0.0113 - accuracy: 0.9961 - val_loss: 1.5348 - val_accuracy: 0.8792
Epoch 62/100
121/121 [==============================] - 14s 118ms/step - loss: 0.0084 - accuracy: 0.9966 - val_loss: 1.2262 - val_accuracy: 0.9062
Epoch 63/100
121/121 [==============================] - 13s 108ms/step - loss: 0.0144 - accuracy: 0.9953 - val_loss: 1.2652 - val_accuracy: 0.8698
Epoch 64/100
121/121 [==============================] - 13s 105ms/step - loss: 0.0337 - accuracy: 0.9906 - val_loss: 0.6206 - val_accuracy: 0.9240
Epoch 65/100
121/121 [==============================] - 13s 108ms/step - loss: 0.0165 - accuracy: 0.9930 - val_loss: 0.6269 - val_accuracy: 0.9365
Epoch 66/100
121/121 [==============================] - 12s 96ms/step - loss: 0.0150 - accuracy: 0.9943 - val_loss: 0.8047 - val_accuracy: 0.9167
Epoch 67/100
121/121 [==============================] - 13s 105ms/step - loss: 0.0129 - accuracy: 0.9961 - val_loss: 0.7999 - val_accuracy: 0.9302
Epoch 68/100
121/121 [==============================] - 13s 106ms/step - loss: 0.0149 - accuracy: 0.9948 - val_loss: 1.3542 - val_accuracy: 0.8792
Epoch 69/100
121/121 [==============================] - 14s 114ms/step - loss: 0.0158 - accuracy: 0.9948 - val_loss: 1.0635 - val_accuracy: 0.8990
Epoch 70/100
121/121 [==============================] - 15s 123ms/step - loss: 0.0130 - accuracy: 0.9958 - val_loss: 1.0130 - val_accuracy: 0.8885
Epoch 71/100
121/121 [==============================] - 13s 109ms/step - loss: 0.0067 - accuracy: 0.9969 - val_loss: 0.5603 - val_accuracy: 0.9375
Epoch 72/100
121/121 [==============================] - 13s 108ms/step - loss: 0.0178 - accuracy: 0.9927 - val_loss: 0.6913 - val_accuracy: 0.9146
Epoch 73/100
121/121 [==============================] - 13s 105ms/step - loss: 0.0117 - accuracy: 0.9956 - val_loss: 1.0109 - val_accuracy: 0.9104
Epoch 74/100
121/121 [==============================] - 13s 107ms/step - loss: 0.0138 - accuracy: 0.9953 - val_loss: 1.0024 - val_accuracy: 0.9052
Epoch 75/100
121/121 [==============================] - 13s 108ms/step - loss: 0.0120 - accuracy: 0.9951 - val_loss: 1.2766 - val_accuracy: 0.9000
Epoch 76/100
121/121 [==============================] - 13s 109ms/step - loss: 0.0155 - accuracy: 0.9940 - val_loss: 0.6932 - val_accuracy: 0.9385
Epoch 77/100
121/121 [==============================] - 13s 109ms/step - loss: 0.0135 - accuracy: 0.9956 - val_loss: 0.8739 - val_accuracy: 0.9344
Epoch 78/100
121/121 [==============================] - 13s 109ms/step - loss: 0.0214 - accuracy: 0.9906 - val_loss: 0.7075 - val_accuracy: 0.9260
Epoch 79/100
121/121 [==============================] - 13s 110ms/step - loss: 0.0145 - accuracy: 0.9945 - val_loss: 0.9346 - val_accuracy: 0.9146
Epoch 80/100
121/121 [==============================] - 15s 124ms/step - loss: 0.0085 - accuracy: 0.9958 - val_loss: 0.8332 - val_accuracy: 0.9323
Epoch 81/100
121/121 [==============================] - 17s 138ms/step - loss: 0.0068 - accuracy: 0.9977 - val_loss: 0.9192 - val_accuracy: 0.9312
Epoch 82/100
121/121 [==============================] - 17s 139ms/step - loss: 0.0097 - accuracy: 0.9966 - val_loss: 1.5400 - val_accuracy: 0.8750
Epoch 83/100
121/121 [==============================] - 16s 134ms/step - loss: 0.0149 - accuracy: 0.9945 - val_loss: 0.8571 - val_accuracy: 0.9240
Epoch 84/100
121/121 [==============================] - 13s 110ms/step - loss: 0.0140 - accuracy: 0.9958 - val_loss: 1.1156 - val_accuracy: 0.8844
Epoch 85/100
121/121 [==============================] - 15s 122ms/step - loss: 0.0175 - accuracy: 0.9932 - val_loss: 0.7525 - val_accuracy: 0.9292
Epoch 86/100
121/121 [==============================] - 14s 116ms/step - loss: 0.0070 - accuracy: 0.9974 - val_loss: 0.8151 - val_accuracy: 0.9333
Epoch 87/100
121/121 [==============================] - 16s 130ms/step - loss: 0.0046 - accuracy: 0.9977 - val_loss: 0.9750 - val_accuracy: 0.9240
Epoch 88/100
121/121 [==============================] - 16s 129ms/step - loss: 0.0110 - accuracy: 0.9966 - val_loss: 1.0478 - val_accuracy: 0.9187
Epoch 89/100
121/121 [==============================] - 13s 108ms/step - loss: 0.0116 - accuracy: 0.9966 - val_loss: 0.9943 - val_accuracy: 0.9135
Epoch 90/100
121/121 [==============================] - 13s 108ms/step - loss: 0.0094 - accuracy: 0.9961 - val_loss: 0.8942 - val_accuracy: 0.9167
Epoch 91/100
121/121 [==============================] - 12s 101ms/step - loss: 0.0093 - accuracy: 0.9958 - val_loss: 1.2042 - val_accuracy: 0.8948
Epoch 92/100
121/121 [==============================] - 13s 105ms/step - loss: 0.0162 - accuracy: 0.9953 - val_loss: 0.7625 - val_accuracy: 0.9302
Epoch 93/100
121/121 [==============================] - 12s 100ms/step - loss: 0.0114 - accuracy: 0.9964 - val_loss: 0.9418 - val_accuracy: 0.9260
Epoch 94/100
121/121 [==============================] - 12s 101ms/step - loss: 0.0117 - accuracy: 0.9961 - val_loss: 1.1174 - val_accuracy: 0.9052
Epoch 95/100
121/121 [==============================] - 14s 114ms/step - loss: 0.0167 - accuracy: 0.9945 - val_loss: 0.5836 - val_accuracy: 0.9458
Epoch 96/100
121/121 [==============================] - 13s 105ms/step - loss: 0.0192 - accuracy: 0.9925 - val_loss: 0.6019 - val_accuracy: 0.9448
Epoch 97/100
121/121 [==============================] - 12s 100ms/step - loss: 0.0160 - accuracy: 0.9951 - val_loss: 1.0855 - val_accuracy: 0.9115
Epoch 98/100
121/121 [==============================] - 14s 115ms/step - loss: 0.0152 - accuracy: 0.9945 - val_loss: 0.9914 - val_accuracy: 0.8885
Epoch 99/100
121/121 [==============================] - 15s 121ms/step - loss: 0.0116 - accuracy: 0.9966 - val_loss: 1.2020 - val_accuracy: 0.9219
Epoch 100/100
121/121 [==============================] - 12s 103ms/step - loss: 0.0084 - accuracy: 0.9969 - val_loss: 0.8176 - val_accuracy: 0.9281"""

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


