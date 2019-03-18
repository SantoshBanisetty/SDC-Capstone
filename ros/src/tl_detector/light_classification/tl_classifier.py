from styx_msgs.msg import TrafficLight
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import rospkg
import tensorflow as tf 

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        r = rospkg.RosPack()
        path = r.get_path('tl_detector')
        #print(path)
        self.model = load_model(path + '/traffic_lights_sim.h5')
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        imrs = cv2.resize(image, (256, 256))
        #imrs = image.img_to_array(imrs) 
        imrs = imrs.astype('float32')
        imrs /= 255.0
        imrs = np.expand_dims(imrs, axis = 0)
        with self.graph.as_default():
            result = self.model.predict(imrs)
            print(result)
            pred = np.argmax(result, axis=1)
            print('Predicted Class:' ,pred[0])
            if pred == 0:
                state = TrafficLight.UNKNOWN
            else:
                state = TrafficLight.RED
        return state
