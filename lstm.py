from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from shared import plot_history, load_data, predict

DATA_PATH = "data.json"


def prepare_dataset(test_size, validation_size):

    # učivatanje podataka
    x, y = load_data(DATA_PATH)

    # podela na trening, validacioni i test dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(input_shape):

    # kreiranje modela
    model = keras.Sequential()

    # Dodavanje dva LSTM sloja sa 64 jedinice. Ovaj broj mora biti pozitivan celi broj i označava
    # dimenziju izlaza. Postoje dva tipa rekurentnih slojeva, prvi je "sequence2sequence" gde
    # prosleđujemo kao ulaz sekvencu i kao izlaz dobijamo sekvencu kako bi narednom sloju mogli
    # da prosledimo tu sekvencu kao ulaz. Drugi je "sequence2vector" gde je sekvenca ulaz, ali
    # izlaz nije sekvenca za naredni sloj, već će tek poslednji rekurentni sloj generisati
    # konačan izlaz. Koristićemo prvi tip zato što hoćemo narednom LSTM sloju da prosledimo
    # izlaznu sekvencu iz prvog sloja. Da bi to omogućili postavićemo argument return_sequences
    # na true.
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

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
    input_shape = (inputs_train.shape[1], inputs_train.shape[2])  # 130, 13
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

    # treniranje modela kroz 30 epoha, tj. ciklusa kroz celi trening dataset. Kao argument
    # prosledićemo i batch_size koji označava broj uzoraka obrađenih pre ažuriranja modela
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_validation, targets_validation),
                        batch_size=32, epochs=50)

    # grafički prikaz tačnost i greške po epohama za trening i validacioni dataset
    plot_history(history)

    # evaluacija modela
    test_loss, test_acc = model.evaluate(inputs_test, targets_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # uzimamo uzorak iz test dataset-a za predviđanje
    X_to_predict = inputs_test[100]
    y_to_predict = targets_test[100]

    # predviđanje uzorka
    predict(model, X_to_predict, y_to_predict)
