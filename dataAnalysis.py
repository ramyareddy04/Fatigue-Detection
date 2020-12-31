import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

txt = ""

data = txt.split(" - ")

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
