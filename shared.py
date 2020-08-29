import matplotlib.pyplot as plt
import json
import numpy as np


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y


def plot_history(history):

    fig, axs = plt.subplots(2)

    # grafik tačnosti
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # grafik greške
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def predict(model, x, y):

    # dodavanje dimenzije ulazu zato što u ovom slučaju model.predict()
    # očekuje 4d niz kao ulaz
    x = x[np.newaxis, ...]  # dimenzije niza (1, 130, 13, 1)

    # predviđanje
    prediction = model.predict(x)

    # index sa maximalnom vrednošću
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))
