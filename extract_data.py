import json
import os
import math
import librosa

DATASET_PATH = "genres"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # sec
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # vrednost sa mapping ključem će sadržati nazive 10 žanrova između kojih se
    # vrši predviđanje. Proći ćemo kroz svaki žanr, kroz svaku traku žanra, podeliti
    # svaku traku na 10 segmenata i za svaki segment izdvojiti vektor od 13 MFCCs
    # i njemu odgovarajuću labelu koja označava određeni žanr
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # prolazimo kroz folder svakog žanra
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # proveravamo da li smo na nivou žanr foldera
        if dirpath is not dataset_path:

            # čuvanje naziva foldera žanra u mapping-u
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # obrada svih audio fajlova u određenom žanr folderu
            for f in filenames:

                # učitavanje audio fajla
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # obrada svakog segmenta audio fajla
                for d in range(num_segments):

                    # računanje početka i kraja trenutnog segmenta
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # izvlačenje mfcc-a
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # čuvanje samo mfcc-a sa očekivanim brojem vektora
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # čuvanje podataka u json fajlu
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
