import cv2

def calculateCloseness(prevFeatures, currentFeatures):
    return cv2.compareHist(prevFeatures, currentFeatures, cv2.HISTCMP_BHATTACHARYYA)

def calculatePersonHistograms(x1, y1, x2, y2, frame):
    sub_frame = get_sub_frame_using_bounding_box_results(x1,y1,x2,y2,frame)
    return cv2.calcHist([sub_frame], [0], None, [256], [0, 256])

def get_sub_frame_using_bounding_box_results(x1, y1, x2, y2,frame):
    return frame[int(y1):  int(y2), int(x1): int(x2)]
