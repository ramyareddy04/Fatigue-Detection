import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

txt = """Epoch 1/100
121/121 [==============================] - 8s 70ms/step - loss: 0.7879 - accuracy: 0.6809 - val_loss: 1.6363 - val_accuracy: 0.1833
Epoch 2/100
121/121 [==============================] - 8s 69ms/step - loss: 0.5114 - accuracy: 0.8120 - val_loss: 1.1401 - val_accuracy: 0.5302
Epoch 3/100
121/121 [==============================] - 10s 79ms/step - loss: 0.4439 - accuracy: 0.8393 - val_loss: 1.3519 - val_accuracy: 0.4896
Epoch 4/100
121/121 [==============================] - 9s 73ms/step - loss: 0.3614 - accuracy: 0.8705 - val_loss: 1.0856 - val_accuracy: 0.5729
Epoch 5/100
121/121 [==============================] - 9s 72ms/step - loss: 0.3335 - accuracy: 0.8811 - val_loss: 0.8956 - val_accuracy: 0.6865
Epoch 6/100
121/121 [==============================] - 8s 67ms/step - loss: 0.3039 - accuracy: 0.8908 - val_loss: 0.9280 - val_accuracy: 0.6729
Epoch 7/100
121/121 [==============================] - 8s 68ms/step - loss: 0.2650 - accuracy: 0.9004 - val_loss: 0.8037 - val_accuracy: 0.7260
Epoch 8/100
121/121 [==============================] - 8s 67ms/step - loss: 0.2404 - accuracy: 0.9085 - val_loss: 1.0860 - val_accuracy: 0.6250
Epoch 9/100
121/121 [==============================] - 8s 67ms/step - loss: 0.2128 - accuracy: 0.9191 - val_loss: 0.8839 - val_accuracy: 0.6812
Epoch 10/100
121/121 [==============================] - 8s 68ms/step - loss: 0.1860 - accuracy: 0.9321 - val_loss: 1.0608 - val_accuracy: 0.6313
Epoch 11/100
121/121 [==============================] - 8s 67ms/step - loss: 0.1734 - accuracy: 0.9339 - val_loss: 0.9695 - val_accuracy: 0.7198
Epoch 12/100
121/121 [==============================] - 8s 67ms/step - loss: 0.1631 - accuracy: 0.9368 - val_loss: 0.8900 - val_accuracy: 0.7083
Epoch 13/100
121/121 [==============================] - 8s 67ms/step - loss: 0.1457 - accuracy: 0.9417 - val_loss: 1.1442 - val_accuracy: 0.6812
Epoch 14/100
121/121 [==============================] - 8s 67ms/step - loss: 0.1344 - accuracy: 0.9485 - val_loss: 1.0977 - val_accuracy: 0.7292
Epoch 15/100
121/121 [==============================] - 8s 70ms/step - loss: 0.1165 - accuracy: 0.9542 - val_loss: 1.0751 - val_accuracy: 0.7354
Epoch 16/100
121/121 [==============================] - 9s 72ms/step - loss: 0.1306 - accuracy: 0.9511 - val_loss: 1.4327 - val_accuracy: 0.6573
Epoch 17/100
121/121 [==============================] - 9s 71ms/step - loss: 0.1069 - accuracy: 0.9555 - val_loss: 1.1805 - val_accuracy: 0.7198
Epoch 18/100
121/121 [==============================] - 8s 67ms/step - loss: 0.1203 - accuracy: 0.9558 - val_loss: 2.3390 - val_accuracy: 0.5083
Epoch 19/100
121/121 [==============================] - 8s 66ms/step - loss: 0.1229 - accuracy: 0.9545 - val_loss: 1.4693 - val_accuracy: 0.6708
Epoch 20/100
121/121 [==============================] - 8s 68ms/step - loss: 0.0981 - accuracy: 0.9649 - val_loss: 1.3896 - val_accuracy: 0.7083
Epoch 21/100
121/121 [==============================] - 8s 70ms/step - loss: 0.0807 - accuracy: 0.9664 - val_loss: 1.7256 - val_accuracy: 0.6427
Epoch 22/100
121/121 [==============================] - 10s 79ms/step - loss: 0.1037 - accuracy: 0.9594 - val_loss: 1.5633 - val_accuracy: 0.6792
Epoch 23/100
121/121 [==============================] - 9s 72ms/step - loss: 0.0906 - accuracy: 0.9662 - val_loss: 1.2372 - val_accuracy: 0.7240
Epoch 24/100
121/121 [==============================] - 9s 73ms/step - loss: 0.0664 - accuracy: 0.9745 - val_loss: 1.4258 - val_accuracy: 0.7260
Epoch 25/100
121/121 [==============================] - 9s 71ms/step - loss: 0.0810 - accuracy: 0.9678 - val_loss: 1.6127 - val_accuracy: 0.6531
Epoch 26/100
121/121 [==============================] - 10s 82ms/step - loss: 0.0812 - accuracy: 0.9672 - val_loss: 1.7763 - val_accuracy: 0.6396
Epoch 27/100
121/121 [==============================] - 10s 82ms/step - loss: 0.0861 - accuracy: 0.9667 - val_loss: 1.3627 - val_accuracy: 0.6906
Epoch 28/100
121/121 [==============================] - 8s 70ms/step - loss: 0.0672 - accuracy: 0.9748 - val_loss: 1.9507 - val_accuracy: 0.6979
Epoch 29/100
121/121 [==============================] - 8s 68ms/step - loss: 0.0688 - accuracy: 0.9745 - val_loss: 1.6363 - val_accuracy: 0.7167
Epoch 30/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0600 - accuracy: 0.9778 - val_loss: 2.5886 - val_accuracy: 0.6531
Epoch 31/100
121/121 [==============================] - 7s 60ms/step - loss: 0.0748 - accuracy: 0.9704 - val_loss: 1.9765 - val_accuracy: 0.6938
Epoch 32/100
121/121 [==============================] - 8s 62ms/step - loss: 0.0540 - accuracy: 0.9789 - val_loss: 1.6828 - val_accuracy: 0.7198
Epoch 33/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0796 - accuracy: 0.9714 - val_loss: 1.2434 - val_accuracy: 0.7521
Epoch 34/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0502 - accuracy: 0.9808 - val_loss: 2.1492 - val_accuracy: 0.6615
Epoch 35/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0617 - accuracy: 0.9756 - val_loss: 1.8552 - val_accuracy: 0.7042
Epoch 36/100
121/121 [==============================] - 7s 56ms/step - loss: 0.0616 - accuracy: 0.9761 - val_loss: 1.8356 - val_accuracy: 0.7125
Epoch 37/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0594 - accuracy: 0.9761 - val_loss: 2.4218 - val_accuracy: 0.5844
Epoch 38/100
121/121 [==============================] - 7s 58ms/step - loss: 0.0672 - accuracy: 0.9743 - val_loss: 2.4346 - val_accuracy: 0.6490
Epoch 39/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0615 - accuracy: 0.9763 - val_loss: 2.4111 - val_accuracy: 0.6490
Epoch 40/100
121/121 [==============================] - 7s 56ms/step - loss: 0.0555 - accuracy: 0.9792 - val_loss: 1.9544 - val_accuracy: 0.6771
Epoch 41/100
121/121 [==============================] - 6s 54ms/step - loss: 0.0329 - accuracy: 0.9901 - val_loss: 2.0938 - val_accuracy: 0.7365
Epoch 42/100
121/121 [==============================] - 6s 54ms/step - loss: 0.0453 - accuracy: 0.9849 - val_loss: 2.7787 - val_accuracy: 0.6198
Epoch 43/100
121/121 [==============================] - 6s 53ms/step - loss: 0.0425 - accuracy: 0.9847 - val_loss: 1.8224 - val_accuracy: 0.7437
Epoch 44/100
121/121 [==============================] - 6s 53ms/step - loss: 0.0473 - accuracy: 0.9821 - val_loss: 1.9052 - val_accuracy: 0.7396
Epoch 45/100
121/121 [==============================] - 6s 54ms/step - loss: 0.0548 - accuracy: 0.9771 - val_loss: 2.2464 - val_accuracy: 0.7010
Epoch 46/100
121/121 [==============================] - 6s 52ms/step - loss: 0.0422 - accuracy: 0.9841 - val_loss: 2.1125 - val_accuracy: 0.7167
Epoch 47/100
121/121 [==============================] - 6s 52ms/step - loss: 0.0419 - accuracy: 0.9821 - val_loss: 2.2041 - val_accuracy: 0.7594
Epoch 48/100
121/121 [==============================] - 6s 52ms/step - loss: 0.0531 - accuracy: 0.9795 - val_loss: 2.3811 - val_accuracy: 0.7302
Epoch 49/100
121/121 [==============================] - 6s 49ms/step - loss: 0.0541 - accuracy: 0.9787 - val_loss: 2.2132 - val_accuracy: 0.7115
Epoch 50/100
121/121 [==============================] - 6s 52ms/step - loss: 0.0567 - accuracy: 0.9795 - val_loss: 3.0989 - val_accuracy: 0.6177
Epoch 51/100
121/121 [==============================] - 6s 50ms/step - loss: 0.0553 - accuracy: 0.9779 - val_loss: 2.9287 - val_accuracy: 0.6375
Epoch 52/100
121/121 [==============================] - 6s 50ms/step - loss: 0.0422 - accuracy: 0.9836 - val_loss: 2.8991 - val_accuracy: 0.6615
Epoch 53/100
121/121 [==============================] - 6s 50ms/step - loss: 0.0311 - accuracy: 0.9875 - val_loss: 2.5207 - val_accuracy: 0.7000
Epoch 54/100
121/121 [==============================] - 6s 50ms/step - loss: 0.0410 - accuracy: 0.9852 - val_loss: 1.9891 - val_accuracy: 0.7250
Epoch 55/100
121/121 [==============================] - 6s 50ms/step - loss: 0.0328 - accuracy: 0.9886 - val_loss: 2.5853 - val_accuracy: 0.7052
Epoch 56/100
121/121 [==============================] - 6s 51ms/step - loss: 0.0552 - accuracy: 0.9810 - val_loss: 2.6745 - val_accuracy: 0.6469
Epoch 57/100
121/121 [==============================] - 8s 64ms/step - loss: 0.0418 - accuracy: 0.9847 - val_loss: 2.9661 - val_accuracy: 0.6427
Epoch 58/100
121/121 [==============================] - 8s 62ms/step - loss: 0.0467 - accuracy: 0.9834 - val_loss: 2.8138 - val_accuracy: 0.6667
Epoch 59/100
121/121 [==============================] - 8s 64ms/step - loss: 0.0351 - accuracy: 0.9891 - val_loss: 2.6676 - val_accuracy: 0.6990
Epoch 60/100
121/121 [==============================] - 7s 61ms/step - loss: 0.0520 - accuracy: 0.9810 - val_loss: 2.7348 - val_accuracy: 0.6562
Epoch 61/100
121/121 [==============================] - 7s 58ms/step - loss: 0.0538 - accuracy: 0.9828 - val_loss: 2.5660 - val_accuracy: 0.6354
Epoch 62/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0632 - accuracy: 0.9756 - val_loss: 3.8745 - val_accuracy: 0.5615
Epoch 63/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0541 - accuracy: 0.9821 - val_loss: 2.4346 - val_accuracy: 0.6635
Epoch 64/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0443 - accuracy: 0.9839 - val_loss: 2.7735 - val_accuracy: 0.6719
Epoch 65/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0369 - accuracy: 0.9841 - val_loss: 2.9099 - val_accuracy: 0.6771
Epoch 66/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0388 - accuracy: 0.9862 - val_loss: 2.5933 - val_accuracy: 0.7094
Epoch 67/100
121/121 [==============================] - 7s 58ms/step - loss: 0.0332 - accuracy: 0.9857 - val_loss: 3.2596 - val_accuracy: 0.6427
Epoch 68/100
121/121 [==============================] - 7s 60ms/step - loss: 0.0362 - accuracy: 0.9865 - val_loss: 2.6912 - val_accuracy: 0.6927
Epoch 69/100
121/121 [==============================] - 7s 58ms/step - loss: 0.0377 - accuracy: 0.9854 - val_loss: 3.0059 - val_accuracy: 0.6708
Epoch 70/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0293 - accuracy: 0.9888 - val_loss: 2.7677 - val_accuracy: 0.7021
Epoch 71/100
121/121 [==============================] - 7s 60ms/step - loss: 0.0449 - accuracy: 0.9847 - val_loss: 2.3436 - val_accuracy: 0.7240
Epoch 72/100
121/121 [==============================] - 7s 60ms/step - loss: 0.0323 - accuracy: 0.9873 - val_loss: 2.6735 - val_accuracy: 0.6979
Epoch 73/100
121/121 [==============================] - 8s 66ms/step - loss: 0.0369 - accuracy: 0.9849 - val_loss: 2.6106 - val_accuracy: 0.6875
Epoch 74/100
121/121 [==============================] - 8s 63ms/step - loss: 0.0377 - accuracy: 0.9857 - val_loss: 3.3229 - val_accuracy: 0.6740
Epoch 75/100
121/121 [==============================] - 7s 59ms/step - loss: 0.0288 - accuracy: 0.9893 - val_loss: 3.6279 - val_accuracy: 0.6250
Epoch 76/100
121/121 [==============================] - 7s 56ms/step - loss: 0.0211 - accuracy: 0.9906 - val_loss: 2.4539 - val_accuracy: 0.7437
Epoch 77/100
121/121 [==============================] - 7s 59ms/step - loss: 0.0236 - accuracy: 0.9893 - val_loss: 3.1836 - val_accuracy: 0.7042
Epoch 78/100
121/121 [==============================] - 7s 57ms/step - loss: 0.0408 - accuracy: 0.9854 - val_loss: 2.6991 - val_accuracy: 0.6823
Epoch 79/100
121/121 [==============================] - 7s 55ms/step - loss: 0.0380 - accuracy: 0.9860 - val_loss: 3.6102 - val_accuracy: 0.6573
Epoch 80/100
121/121 [==============================] - 8s 62ms/step - loss: 0.0581 - accuracy: 0.9810 - val_loss: 2.9891 - val_accuracy: 0.6344
Epoch 81/100
121/121 [==============================] - 9s 72ms/step - loss: 0.0453 - accuracy: 0.9827 - val_loss: 2.7462 - val_accuracy: 0.6854
Epoch 82/100
121/121 [==============================] - 10s 79ms/step - loss: 0.0334 - accuracy: 0.9867 - val_loss: 3.3637 - val_accuracy: 0.6687
Epoch 83/100
121/121 [==============================] - 8s 67ms/step - loss: 0.0386 - accuracy: 0.9870 - val_loss: 4.0958 - val_accuracy: 0.6396
Epoch 84/100
121/121 [==============================] - 9s 74ms/step - loss: 0.0451 - accuracy: 0.9841 - val_loss: 3.7109 - val_accuracy: 0.6510
Epoch 85/100
121/121 [==============================] - 8s 62ms/step - loss: 0.0345 - accuracy: 0.9867 - val_loss: 2.4139 - val_accuracy: 0.7396
Epoch 86/100
121/121 [==============================] - 8s 66ms/step - loss: 0.0278 - accuracy: 0.9906 - val_loss: 2.6254 - val_accuracy: 0.7156
Epoch 87/100
121/121 [==============================] - 8s 62ms/step - loss: 0.0265 - accuracy: 0.9893 - val_loss: 3.3617 - val_accuracy: 0.6448
Epoch 88/100
121/121 [==============================] - 8s 69ms/step - loss: 0.0303 - accuracy: 0.9886 - val_loss: 3.6221 - val_accuracy: 0.6552
Epoch 89/100
121/121 [==============================] - 8s 67ms/step - loss: 0.0234 - accuracy: 0.9909 - val_loss: 3.7113 - val_accuracy: 0.6406
Epoch 90/100
121/121 [==============================] - 7s 59ms/step - loss: 0.0424 - accuracy: 0.9847 - val_loss: 3.2697 - val_accuracy: 0.6562
Epoch 91/100
121/121 [==============================] - 8s 65ms/step - loss: 0.0370 - accuracy: 0.9852 - val_loss: 2.8832 - val_accuracy: 0.6760
Epoch 92/100
121/121 [==============================] - 7s 58ms/step - loss: 0.0408 - accuracy: 0.9860 - val_loss: 3.2403 - val_accuracy: 0.6656
Epoch 93/100
121/121 [==============================] - 7s 56ms/step - loss: 0.0296 - accuracy: 0.9886 - val_loss: 3.4471 - val_accuracy: 0.6812
Epoch 94/100
121/121 [==============================] - 7s 59ms/step - loss: 0.0210 - accuracy: 0.9917 - val_loss: 4.3944 - val_accuracy: 0.6385
Epoch 95/100
121/121 [==============================] - 7s 56ms/step - loss: 0.0242 - accuracy: 0.9927 - val_loss: 3.8136 - val_accuracy: 0.6667
Epoch 96/100
121/121 [==============================] - 7s 55ms/step - loss: 0.0197 - accuracy: 0.9935 - val_loss: 3.8204 - val_accuracy: 0.6708
Epoch 97/100
121/121 [==============================] - 7s 56ms/step - loss: 0.0238 - accuracy: 0.9912 - val_loss: 3.9801 - val_accuracy: 0.6740
Epoch 98/100
121/121 [==============================] - 7s 58ms/step - loss: 0.0234 - accuracy: 0.9925 - val_loss: 3.9730 - val_accuracy: 0.6250
Epoch 99/100
121/121 [==============================] - 7s 56ms/step - loss: 0.0276 - accuracy: 0.9891 - val_loss: 3.2093 - val_accuracy: 0.6594
Epoch 100/100
121/121 [==============================] - 8s 64ms/step - loss: 0.0356 - accuracy: 0.9862 - val_loss: 3.6860 - val_accuracy: 0.6729"""

data = txt.split(" - ")

loss = []
accuracy = []
val_loss = []
val_accuracy = []
for i in data:
    if i.find("loss: ") and (i.find("val_loss: ")):
        print(i)
        loss.append(i[6:])
    if i.find("val_loss: "):
        val_loss.append(i[10:])
        #print(i)
    if i.find("accuracy: ") and not i.find("val_accuracy: "):
        accuracy.append(i[11:])

print(len(loss), len(val_loss))


print(data)

# test6Loss = [0.1896,0.0628,0.0454,0.0384,0.0335, 0.0300,0.0274,0.0254,0.0226,0.0206,0.0198,0.0186,0.0184,0.0174,0.0163,0.0144,0.0144,0.0138,0.0138,0.0126]
# test6Accuracy = [0.3591,0.8145,0.8711,0.8915,0.9103,0.9204,0.9281,0.9366,0.9422,0.9475,0.9493,0.9546,0.9550,0.9573,0.9602,0.9658,0.9576,0.9681,0.9686,0.9716]
# test6val_loss = [0.7108, 1.2496, 1.4878, 2.0569, 2.0589, 2.1557, 2.7852, 2.6990, 2.7173, 2.6202, 2.8806, 3.2430, 2.8159, 2.8363, 3.4131, 3.4911, 3.2052, 3.2917, 3.417, 3.7565]
# test6val_accuracy = np.array([0.0145, 0.0161, 0.0155, 0.0128, 0.0158, 0.0118, 0.0143, 0.0149, 0.0139, 0.0144, 0.0153, 0.0166, 0.0143, 0.0174, 0.0122, 0.0184, 0.0150, 0.0153, 0.0109, 0.0116])
# df = pd.DataFrame(test6Loss ,columns=["Loss"])
# print(df.loc[:, "Loss"])
#
# plt.plot(test6Loss, 'g', label='Training loss')
# plt.plot(test6val_loss, 'b', label='validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.plot(test6Accuracy, 'g', label='Training accuracy')
# plt.plot(test6val_accuracy, 'b', label='validation accuracy')
# plt.title('Training and Validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
#
# print(test6val_accuracy.shape)
