# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

from k2darknet import darknet as dn

dn.set_gpu(0)
net = dn.load_net(b'cfg/yolov3.cfg', b'yolov3.weights', 0)
meta = dn.load_meta(b'cfg/coco.data')
r = dn.detect(net, meta, b'data/kite.jpg')
print(r)

# And then down here you could detect a lot more images like:
r = dn.detect(net, meta, b'data/eagle.jpg')
print(r)
r = dn.detect(net, meta, b'data/giraffe.jpg')
print(r)
r = dn.detect(net, meta, b'data/horses.jpg')
print(r)
r = dn.detect(net, meta, b'data/person.jpg')
print(r)
