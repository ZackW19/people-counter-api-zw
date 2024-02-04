#
# Wersja na 4
#
from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from io import BytesIO  # Importuje klasę BytesIO z modułu io, która pozwala na operacje na danych binarnych w formie bufora bajtów.

app = Flask(__name__)


def count_people(image):
    # Skonfiguruj detektor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Wczytaj obraz z danych przesłanych w formie binarnej
    nparr = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Wykryj ludzi na zdjęciu
    boxes, weights = hog.detectMultiScale(img, winStride=(6, 6))

    # Parametry detektora HOG:
    # winStride=(6, 6): Przesunięcie okna detekcji w poziomie i pionie podczas przeszukiwania obrazu. Im mniejsza tym więcej wykrywa
    # opcjonalnie: padding=(4, 4): Ilość pikseli dodanych na obrzeżach okna detekcji w celu zwiększenia dokładności detekcji.
    # opcjonalnie: scale=1.03: Skalowanie obrazu podczas kolejnych iteracji detekcji. Wyższa wartość oznacza zmniejszenie obrazu w każdej iteracji.
    #
    # boxes - lista prostokątnych obszarów, w których detektor znalazł obiekty (ludzi)
    # weights - lista wag przypisanych do każdego z obszarów, wskazujących na pewność detekcji

    # liczba osób
    return len(boxes)


@app.route('/url', methods=['GET'])  # App Routing mapuje url do funkcji obsugujacej logikę ukrytą pod tym url. Obsługuje jedynie żądania HTTP typu GET.
def detect_people():  # Funkcja obsługująca żądania na endpointcie
    try:
        # Sprawdź czy podano parametr url
        image_url = request.args.get('url')
        if not image_url:
            return jsonify({'error': 'Brak parametru url'}), 400

        # Wczytaj obraz z URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Brak obrazu w podanym URL'}), 400

        # Analizuj osoby na zdjęciu
        num_people = count_people(BytesIO(response.content))
        return jsonify({'num_people': num_people})

    except Exception as e:
        return jsonify({'error': f'Błąd przetwarzania obrazu: {str(e)}'}), 500


if __name__ == '__main__':  # Warunek sprawdzający, czy skrypt jest uruchamiany jako główny program (a nie importowany jako moduł)
    app.run(debug=True, port=8000)  # Uruchamia aplikację Flask, włączając tryb debugowania. Ustawienie portu aplikacji na 8000
