import k2darknet.darknet as dn


class Detector():
    def __init__(self):
        self.net = dn.load_net(b'cfg/yolov3.cfg', b'yolov3.weights', 0)
        self.meta = dn.load_meta(b'cfg/coco.data')

    def detect(self, filename):
        r = dn.detect(self.net, self.meta, filename.encode())

        results = []
        for entry in r:
            result = {}
            results.append(result)
            result['class'] = entry[0].decode()
            result['probability'] = entry[1]
            result['center'] = (entry[2][0], entry[2][1])
            result['width'] = entry[2][2]
            result['height'] = entry[2][3]

        return results
