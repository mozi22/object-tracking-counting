from pydarknet import Detector, Image


class ObjectDetection:

    def __init__(self):

        root = '/home/muazzam/mywork/object-tracking-counting'

        configPath  = root + '/brain/yolov3.cfg'
        weightsPath = root + '/brain/yolov3.weights'
        classesPath = root + '/brain/coco.data'

        self.net = Detector(bytes(configPath, encoding="utf-8"), 
                        bytes(weightsPath, encoding="utf-8"), 0,
                        bytes(classesPath, encoding="utf-8"))


    def run(self, frame):
        dark_frame = Image(frame)
        results = self.net.detect(dark_frame)

        # print(serializedResult)
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        # 	break
        #del dark_frame

        return results
