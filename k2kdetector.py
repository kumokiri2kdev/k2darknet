from k2darknet import k2kdarknet as dn
import cv2

FILE_NAME = 'data/kite.jpg'
OUT_FILE_NAME = 'data/detected.jpg'

detector = dn.Detector()

results = detector.detect(FILE_NAME)

print(results)

img = cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)

for result in results:
    x, y = [i for i in result['center']]
    h_offset = result['width'] / 2
    v_offset = result['height'] / 2
    cv2.rectangle(img, (int(x - h_offset), int(y - v_offset)),
                  (int(x + h_offset), int(y + v_offset)), (0, 255, 0))


cv2.imwrite(OUT_FILE_NAME, img)


