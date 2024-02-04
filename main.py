#
# Wersja na 3
#

from flask import Flask
from flask_restful import Resource, Api
import cv2

app = Flask(__name__)
api = Api(app)

# konfiguracja detektora osób
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# Rozpoznawanie z pliku wskazanego statycznie w zmiennej
class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('images/ludzie2.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(4, 4))
        # winStride zmiana tego param. zwieksza/zmnijsza dokladnsosc
        # im mniejsza tym dokladniej lczy. przesuniecie okienka

        return {'count': len(boxes)}
        # return {'count': 2}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(HelloWorld, '/test')  # endpoint test
api.add_resource(PeopleCounter, '/img')  # endpoint img

if __name__ == '__main__':
    app.run(debug=True)
