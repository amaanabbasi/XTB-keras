from model import finetuning_model, PlotLossAcc, DataGenerator
import keras


def print_stats(model, epochs, lr):
    print("epochs: {}, learning rate: {}".format(epochs, lr))
    print()
    print(model.summary())
    
    
d = DataGenerator()
d.train_test_generator()

epochs = 1
lr = 1.1

Adam = keras.optimizers.Adam(lr=lr)

model = finetuning_model()

print_stats(model, epochs, lr)
model.compile(optimizer=Adam,loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit_generator(generator=d.train_generator,
                              steps_per_epoch=d.step_size_train,
                              validation_data=d.valid_generator,
                              validation_steps=d.step_size_valid,
                              epochs=epochs)

model.save('models/VGG16-{}-{}-adam.h5'.format(epochs, lr,))
h = PlotLossAcc(history)
h.plot_loss_acc()
h.save_history()