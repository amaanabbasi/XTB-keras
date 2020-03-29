from model import finetuning_model, PlotLossAcc, DataGenerator
import keras


def print_stats(model, epochs, lr):
    print("epochs: {}, learning rate: {}".format(epochs, lr))
    print()
    print(model.summary())
    
    
d = DataGenerator()
d.train_valid_generator()

epochs = 50
lr = 0.001

Adam = keras.optimizers.Adam(lr=lr)

model = finetuning_model(batch_normalization=0)

print_stats(model, epochs, lr)
model.compile(optimizer=Adam,loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit_generator(generator=d.train_generator,
                              steps_per_epoch=d.step_size_train,
                              validation_data=d.valid_generator,
                              validation_steps=d.step_size_valid,
                              epochs=epochs)

model.save('models/VGG16-batch-normalizations-{}-{}-adam.h5'.format(epochs, lr,))
h = PlotLossAcc(history)
h.plot_loss_acc()
h.save_history()

# with normalization
# 62/62 [==============================] - 45s 730ms/step - loss: 1.9853 - acc: 0.5111 - val_loss: 0.6828 - val_acc: 0.5583

# Epoch 1/5
# 62/62 [==============================] - 46s 749ms/step - loss: 0.7437 - acc: 0.6436 - val_loss: 0.5541 - val_acc: 0.7667
# Epoch 2/5
# 62/62 [==============================] - 47s 766ms/step - loss: 0.5560 - acc: 0.7323 - val_loss: 0.4860 - val_acc: 0.7583
# Epoch 3/5
# 62/62 [==============================] - 47s 766ms/step - loss: 0.5351 - acc: 0.7404 - val_loss: 0.4841 - val_acc: 0.7500
# Epoch 4/5
# 62/62 [==============================] - 47s 765ms/step - loss: 0.5572 - acc: 0.7233 - val_loss: 0.4592 - val_acc: 0.7750
# Epoch 5/5
# 62/62 [==============================] - 48s 768ms/step - loss: 0.5014 - acc: 0.7712 - val_loss: 0.4972 - val_acc: 0.7500

#Epoch 1/50
# 62/62 [==============================] - 44s 717ms/step - loss: 0.7281 - acc: 0.6583 - val_loss: 0.5076 - val_acc: 0.8083
# Epoch 2/50
# 62/62 [==============================] - 48s 767ms/step - loss: 0.5709 - acc: 0.7233 - val_loss: 0.4530 - val_acc: 0.7667
# Epoch 3/50
# 62/62 [==============================] - 48s 776ms/step - loss: 0.5587 - acc: 0.7319 - val_loss: 0.4969 - val_acc: 0.7250
# Epoch 4/50
# 62/62 [==============================] - 48s 772ms/step - loss: 0.5304 - acc: 0.7485 - val_loss: 0.5205 - val_acc: 0.7750
# Epoch 5/50
# 62/62 [==============================] - 48s 772ms/step - loss: 0.5202 - acc: 0.7445 - val_loss: 0.5299 - val_acc: 0.7917
# Epoch 6/50
# 62/62 [==============================] - 48s 771ms/step - loss: 0.4901 - acc: 0.7682 - val_loss: 0.5120 - val_acc: 0.7667
# Epoch 7/50
# 62/62 [==============================] - 48s 775ms/step - loss: 0.4888 - acc: 0.7722 - val_loss: 0.5028 - val_acc: 0.7833
# Epoch 8/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4795 - acc: 0.7752 - val_loss: 0.5001 - val_acc: 0.7917
# Epoch 9/50
# 62/62 [==============================] - 47s 765ms/step - loss: 0.4853 - acc: 0.7767 - val_loss: 0.5357 - val_acc: 0.7833
# Epoch 10/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4681 - acc: 0.7782 - val_loss: 0.5159 - val_acc: 0.8083
# Epoch 11/50
# 62/62 [==============================] - 47s 765ms/step - loss: 0.4754 - acc: 0.7893 - val_loss: 0.5142 - val_acc: 0.7833
# Epoch 12/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4585 - acc: 0.7863 - val_loss: 0.5002 - val_acc: 0.8000
# Epoch 13/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4616 - acc: 0.7813 - val_loss: 0.5221 - val_acc: 0.7917
# Epoch 14/50
# 62/62 [==============================] - 48s 767ms/step - loss: 0.4382 - acc: 0.7974 - val_loss: 0.5545 - val_acc: 0.7833
# Epoch 15/50
# 62/62 [==============================] - 48s 769ms/step - loss: 0.4298 - acc: 0.8065 - val_loss: 0.5152 - val_acc: 0.7750
# Epoch 16/50
# 62/62 [==============================] - 47s 764ms/step - loss: 0.4327 - acc: 0.7979 - val_loss: 0.5201 - val_acc: 0.8250
# Epoch 17/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4191 - acc: 0.8115 - val_loss: 0.5371 - val_acc: 0.8000
# Epoch 18/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4257 - acc: 0.8090 - val_loss: 0.6322 - val_acc: 0.7750
# Epoch 19/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4114 - acc: 0.8095 - val_loss: 0.5049 - val_acc: 0.7667
# Epoch 20/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4091 - acc: 0.8165 - val_loss: 0.5775 - val_acc: 0.7917
# Epoch 21/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4007 - acc: 0.8145 - val_loss: 0.5459 - val_acc: 0.7750
# Epoch 22/50
# 62/62 [==============================] - 48s 767ms/step - loss: 0.3796 - acc: 0.8312 - val_loss: 0.5082 - val_acc: 0.7667
# Epoch 23/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3878 - acc: 0.8311 - val_loss: 0.6008 - val_acc: 0.7667
# Epoch 24/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3824 - acc: 0.8236 - val_loss: 0.5828 - val_acc: 0.7833
# Epoch 25/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3616 - acc: 0.8357 - val_loss: 0.5560 - val_acc: 0.7583
# Epoch 26/50
# 62/62 [==============================] - 47s 765ms/step - loss: 0.3715 - acc: 0.8317 - val_loss: 0.6430 - val_acc: 0.7417
# Epoch 27/50
# 62/62 [==============================] - 47s 764ms/step - loss: 0.3936 - acc: 0.8140 - val_loss: 0.5439 - val_acc: 0.7750
# Epoch 28/50
# 62/62 [==============================] - 48s 768ms/step - loss: 0.3523 - acc: 0.8372 - val_loss: 0.5183 - val_acc: 0.7833
# Epoch 29/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3504 - acc: 0.8488 - val_loss: 0.6470 - val_acc: 0.7417
# Epoch 30/50
# 62/62 [==============================] - 47s 765ms/step - loss: 0.3587 - acc: 0.8503 - val_loss: 0.5822 - val_acc: 0.7833
# Epoch 31/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3639 - acc: 0.8387 - val_loss: 0.5582 - val_acc: 0.8000
# Epoch 32/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3414 - acc: 0.8493 - val_loss: 0.6292 - val_acc: 0.7417
# Epoch 33/50
# 62/62 [==============================] - 47s 764ms/step - loss: 0.3655 - acc: 0.8307 - val_loss: 0.5661 - val_acc: 0.7750
# Epoch 34/50
# 62/62 [==============================] - 48s 766ms/step - loss: 0.3577 - acc: 0.8307 - val_loss: 0.5991 - val_acc: 0.7750
# Epoch 35/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3408 - acc: 0.8508 - val_loss: 0.6000 - val_acc: 0.7917
# Epoch 36/50
# 62/62 [==============================] - 47s 764ms/step - loss: 0.3191 - acc: 0.8624 - val_loss: 0.6072 - val_acc: 0.7917
# Epoch 37/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3231 - acc: 0.8538 - val_loss: 0.5455 - val_acc: 0.7667
# Epoch 38/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3340 - acc: 0.8523 - val_loss: 0.7046 - val_acc: 0.7750
# Epoch 39/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3188 - acc: 0.8669 - val_loss: 0.6191 - val_acc: 0.7667
# Epoch 40/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3059 - acc: 0.8604 - val_loss: 0.9015 - val_acc: 0.7000
# Epoch 41/50
# 62/62 [==============================] - 48s 772ms/step - loss: 0.3109 - acc: 0.8609 - val_loss: 0.5321 - val_acc: 0.7833
# Epoch 42/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3271 - acc: 0.8579 - val_loss: 0.5730 - val_acc: 0.7667
# Epoch 43/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3210 - acc: 0.8579 - val_loss: 0.7201 - val_acc: 0.7583
# Epoch 44/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.2734 - acc: 0.8780 - val_loss: 0.5425 - val_acc: 0.8083
# Epoch 45/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3241 - acc: 0.8599 - val_loss: 0.6109 - val_acc: 0.7750
# Epoch 46/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.3094 - acc: 0.8619 - val_loss: 0.5803 - val_acc: 0.7750
# Epoch 47/50
# 62/62 [==============================] - 48s 767ms/step - loss: 0.3207 - acc: 0.8533 - val_loss: 0.5460 - val_acc: 0.7583
# Epoch 48/50
# 62/62 [==============================] - 47s 764ms/step - loss: 0.3015 - acc: 0.8710 - val_loss: 0.6051 - val_acc: 0.7750
# Epoch 49/50
# 62/62 [==============================] - 47s 765ms/step - loss: 0.2616 - acc: 0.8821 - val_loss: 0.6163 - val_acc: 0.7750
# Epoch 50/50
# 62/62 [==============================] - 47s 763ms/step - loss: 0.2942 - acc: 0.8700 - val_loss: 0.6376 - val_acc: 0.7667

# Epoch 1/10
# 62/62 [==============================] - 47s 760ms/step - loss: 0.7857 - acc: 0.6442 - val_loss: 0.5231 - val_acc: 0.7250
# Epoch 2/10
# 62/62 [==============================] - 47s 764ms/step - loss: 0.5655 - acc: 0.7183 - val_loss: 0.4893 - val_acc: 0.7500
# Epoch 3/10
# 62/62 [==============================] - 48s 768ms/step - loss: 0.5398 - acc: 0.7364 - val_loss: 0.5678 - val_acc: 0.7250
# Epoch 4/10
# 62/62 [==============================] - 47s 764ms/step - loss: 0.5334 - acc: 0.7430 - val_loss: 0.4964 - val_acc: 0.7750
# Epoch 5/10
# 62/62 [==============================] - 47s 765ms/step - loss: 0.5023 - acc: 0.7606 - val_loss: 0.4781 - val_acc: 0.8083
# Epoch 6/10
# 62/62 [==============================] - 48s 772ms/step - loss: 0.5001 - acc: 0.7707 - val_loss: 0.4857 - val_acc: 0.7833
# Epoch 7/10
# 62/62 [==============================] - 48s 768ms/step - loss: 0.5178 - acc: 0.7515 - val_loss: 0.5456 - val_acc: 0.7500
# Epoch 8/10
# 62/62 [==============================] - 47s 763ms/step - loss: 0.5029 - acc: 0.7626 - val_loss: 0.4779 - val_acc: 0.7833
# Epoch 9/10
# 62/62 [==============================] - 47s 763ms/step - loss: 0.5136 - acc: 0.7697 - val_loss: 0.5048 - val_acc: 0.7833
# Epoch 10/10
# 62/62 [==============================] - 47s 763ms/step - loss: 0.4876 - acc: 0.7807 - val_loss: 0.5782 - val_acc: 0.7417


# without normalization
# 62/62 [==============================] - 43s 693ms/step - loss: 7.9056 - acc: 0.4990 - val_loss: 8.0151 - val_acc: 0.5000

# Epoch 1/5
# 62/62 [==============================] - 43s 693ms/step - loss: 0.7180 - acc: 0.5832 - val_loss: 0.5085 - val_acc: 0.7667
# Epoch 2/5
# 62/62 [==============================] - 47s 758ms/step - loss: 0.5815 - acc: 0.7026 - val_loss: 0.5356 - val_acc: 0.7417
# Epoch 3/5
# 62/62 [==============================] - 48s 766ms/step - loss: 0.5348 - acc: 0.7439 - val_loss: 0.4932 - val_acc: 0.7417
# Epoch 4/5
# 62/62 [==============================] - 47s 764ms/step - loss: 0.5142 - acc: 0.7550 - val_loss: 0.4965 - val_acc: 0.7417
# Epoch 5/5
# 62/62 [==============================] - 47s 763ms/step - loss: 0.5062 - acc: 0.7697 - val_loss: 0.4892 - val_acc: 0.7500
