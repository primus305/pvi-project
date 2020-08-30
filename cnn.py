import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from shared import plot_history, predict, load_data

DATA_PATH = "data.json"


def prepare_dataset(test_size, validation_size):

    # učitavanje podataka
    x, y = load_data(DATA_PATH)

    # podela na trening, validacioni i test dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    # dodavanje ose ulaznom set-u
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(input_shape):

    # Kreiranje sekvencijalnog modela
    model = keras.Sequential()

    # Prvi konvolucioni sloj sa 32 filtera (krenela) dizmenzije 3 x 3. aktivaciona funkcija
    # za ovaj sloj, a to je u ovom slučaju ispravljena linearna funkcija (Rectified Linear
    # Unit - ReLU). Ako je ulaz u ReLU manji od nule izlaz će biti nula, ako je veći ili jednak
    # nuli onda će izlaz biti jednak tom broju. Ulazni konvolucioni sloj očekuje trodimenzionalni
    # ulaz.
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # Dodajemo "Max pooling" sloj koji smanjuje broj uzoraka ali ostavlja najbitnije informacije.
    # "Pooling" matrica će biti dimenzije 3 x 3, a pomeranje (korak) matrice je 2 i vertikalno i
    # horizontalno. Padding "same" će dodati nule ulaznoj matrici ukoliko zbog koraka "pooling"
    # matrica ne može da obuhvati željenu dimenziju
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    # Ovim slojem povećavamo brzinu i stabilnost naše mreže tako što normalizujemo ulaze
    model.add(keras.layers.BatchNormalization())

    # Drugi konvolucioni sloj
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Treći konvolucioni sloj
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Kao izlaz iz konvolucionog sloja očekujemo dvodimenzionalni niz i želimo zalepiti njegove
    # dimenzije i napraviti jednodimenzionalni niz.
    model.add(keras.layers.Flatten())
    # Ovaj sloj je potpuno povezan sloj, povezuje sve neurone iz prethodnog sloja sa neuronima
    # iz ovog sloja. Prvi argument predstavlja broj neurona u sloju, drugi je aktivaciona funkcija
    # za ovaj sloj, a to je u ovom slučaju opet ReLU. Dropout je tehnika za rešavanje "overfitting"
    # problema, koja nasumično bira neurone koje će izbaciti tokom treninga. Dropout funkciji
    # prosleđujemo verovatnoću izbacivanja kao argument.
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))

    # Izlazni sloj, koji je takođe potpuno povezan sloj kao i ostali "Dense" slojevi.
    # Broj neuorana će biti 10 zato što imamo 10 kategorija, odnosno muzičkih žanrova
    # između kojih želimo da vršimo predviđanje. Aktivaciona funkcija je softmax čiji je
    # rezultat raspodela verovatnoća po ovim kategorijama
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":

    # podela dataset-a na trening, validacioni i test dataset. Ideja validacionog dela je
    # mogućnost prilagođavanja modela rezultatima metrika koje dobijamo iz ovog dela.
    inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_dataset(0.25, 0.2)

    # kreiranje modela
    input_shape = (inputs_train.shape[1], inputs_train.shape[2], 1)
    model = build_model(input_shape)

    # Kompajliranje modela. Optimizator koji ćemo koristiti je Adam. On predstavlja stohastičku
    # metodu gradijentog spusta. Ukoliko imamo dve ili više klasa i ukoliko su labele brojevi
    # preporuka je da se koristi sparse_categorical_crossentropy kao funkcija greške. Pratićemo
    # tačnost kao metriku
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # treniranje modela kroz 50 epoha, tj. ciklusa kroz celi trening dataset. Kao argument
    # prosledićemo i batch_size koji označava broj uzoraka obrađenih pre ažuriranja modela
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_validation, targets_validation),
                        batch_size=32, epochs=50)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluacija modela
    test_loss, test_acc = model.evaluate(inputs_test, targets_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # uzimamo uzorak iz test dataset-a za predviđanje
    X_to_predict = inputs_test[100]
    y_to_predict = targets_test[100]

    # predviđanje uzorka
    predict(model, X_to_predict, y_to_predict)
