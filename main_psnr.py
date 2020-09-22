from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import prepare_data as pd
import numpy as np
import math
from keras import backend as K

def PSNRLoss(y_true, y_pred):
    PSNR = -10.*K.log(K.mean(K.square(y_pred - y_true)))/K.log(10.)
    return PSNR

def model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(nb_filter=64, nb_row=9, nb_col=9, init='glorot_uniform',
                     activation='relu', border_mode='valid', bias=True, input_shape=(32, 32, 1)))
    SRCNN.add(Conv2D(nb_filter=32, nb_row=1, nb_col=1, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True))
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                     activation='linear', border_mode='valid', bias=True))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=[PSNRLoss])

    return SRCNN

def predict_model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(nb_filter=64, nb_row=9, nb_col=9, init='glorot_uniform',
                     activation='relu', border_mode='valid', bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(nb_filter=32, nb_row=1, nb_col=1, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True))
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                     activation='linear', border_mode='valid', bias=True))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=[PSNRLoss])
    return SRCNN

def train():
    srcnn_model = model()
    print(srcnn_model.summary())
    data, label = pd.read_training_data("./crop_train.h5") 
    val_data, val_label = pd.read_training_data("./test.h5")

    checkpoint = ModelCheckpoint("SRCNN_scale3.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    h = srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, nb_epoch=200, verbose=0)

    loss_history = h.history['PSNRLoss']
    val_loss_history = h.history['val_PSNRLoss']

    PSNR_history = np.array(loss_history)
    PSNR_val_history = np.array(val_loss_history)
    np.savetxt("butterfly_scale3.txt", PSNR_history, delimiter=",")
    np.savetxt("butterfly_val_scale3.txt", PSNR_val_history, delimiter=",")
    print(h.history.keys())

def predict():
    srcnn_model = predict_model()
    srcnn_model.load_weights("SRCNN_scale3.h5") #("3051crop_weight_200.h5")
    IMG_NAME = "/home/huixin/DLProject_SRCNN/SRCNN-keras/Test/Set5/butterfly_GT.bmp"#IMAGE PATH
    INPUT_NAME = "butterfly_bicubic_scale3.jpg"
    OUTPUT_NAME = "butterfly_SRCNN_scale3.jpg"

    #Bicubic computation
    import cv2
    scale = 3
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    Y_img = cv2.resize(img[:, :, 0], (shape[1] // scale, shape[0] // scale), cv2.INTER_CUBIC) #INTEGER DIVISION
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(INPUT_NAME, img)

    #SRCNN computation
    Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

    print("bicubic:")
    print(cv2.PSNR(im1, im2))
    print("SRCNN:")
    print(cv2.PSNR(im1, im3))

if __name__ == "__main__":
    train()
    predict()
