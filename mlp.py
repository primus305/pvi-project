from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from shared import load_data, plot_history

DATA_PATH = "data.json"


if __name__ == "__main__":
    # učitavanje obrađenih podataka, inputs predstavlja numpy niz čiji su elementi
    # MFCC vektori, targets je takođe numpy niz čiji su elementi labele koje
    # predstavljaju žanrove za određeni MFCC vektor
    inputs, targets = load_data(DATA_PATH)

    # podela dataset-a na trening i test dataset
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2)

    # kreiranje modela, u ovom slučaju sekvencijalnog modela, zato što ulazi putuju
    # od ulaznog do izlaznog sloja sekvencijalno
    model = keras.Sequential([

        # Ulazni sloj koji će "zalepiti" dimenzije ulaza i od dvodimenzionalnog niza napraviti
        # jednodimenzionalni. Već smo pomenuli da je inputs niz MFCC vektora, ali on zapravo
        # predstavlja trodimenzionalni niz zato što su njegovi elementi MFCC vektori po
        # segmentima za svaku traku.
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # Prvi "Dense" sloj. Ovaj sloj je potpuno povezan sloj, povezuje sve neurone iz
        # prethodnog sloja sa neuronima iz ovog sloja. Prvi argument predstavlja broj neurona
        # u sloju, drugi je aktivaciona funkcija za ovaj sloj, a to je u ovom slučaju
        # ispravljena linearna funkcija (Rectified Linear Unit - ReLU). Ako je ulaz u ReLU manji
        # od nule izlaz će biti nula, ako je veći ili jednak nuli onda će izlaz biti jednak tom
        # broju. Upotrebljavamo l2 regularizator koji minimizuje kvadrate težinskih koeficijenata
        # kako bi sprečili "overfitting" problem. Dropout je takođe tehnika za rešavanje ovog
        # problema, koja nasumično bira neurone koje će izbaciti tokom treninga. Dropout
        # funkciji prosleđujemo verovatnoću izbacivanja kao argument.
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Drugi "Dense" sloj
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Treci "Dense" sloj
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Izlazni sloj, koji je takođe potpuno povezan sloj kao i ostali "Dense" slojevi.
        # Broj neuorana će biti 10 zato što imamo 10 kategorija, odnosno muzičkih žanrova
        # između kojih želimo da vršimo predviđanje. Aktivaciona funkcija je softmax čiji je
        # rezultat raspodela verovatnoća po ovim kategorijama
        keras.layers.Dense(10, activation='softmax')
    ])

    # Kompajliranje modela. Optimizator koji ćemo koristiti je Adam. On predstavlja stohastičku
    # metodu gradijentog spusta. Ukoliko imamo dve ili više klasa i ukoliko su labele brojevi
    # preporuka je da se koristi sparse_categorical_crossentropy kao funkcija greške. Pratićemo
    # tačnost kao metriku
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # treniranje modela kroz 100 epoha, tj. ciklusa kroz celi trening dataset. Kao argument
    # prosledićemo i batch_size koji označava broj uzoraka obrađenih pre ažuriranja modela
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=100, batch_size=32)

    # evaluacija modela
    model.evaluate(inputs_test, targets_test, verbose=1)

    # grafički prikaz tačnosti i greške po epohama
    plot_history(history)
