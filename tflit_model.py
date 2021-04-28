import tensorflow as tf


class TensorFlowModel:
    def load(self, model_filename, num_threads=None):
        self.interpreter = tf.lite.Interpreter(model_filename,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()

    def resize_input(self, shape):
        if list(self.get_input_shape()) != shape:
            self.interpreter.resize_tensor_input(0, shape)
            self.interpreter.allocate_tensors()

    def get_input_shape(self):
        return self.interpreter.get_input_details()[0]['shape']

    def predict(self, x):
        # assumes one input and one output for now
        self.interpreter.set_tensor(
            self.interpreter.get_input_details()[0]['index'], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(
            self.interpreter.get_output_details()[0]['index'])

if __name__ == '__main__':
    import os
    import numpy as np
    import time
    model = TensorFlowModel()
    model.load(os.path.join(os.getcwd(), 'facenet_tflite/model.tflite'))
    img = np.random.random((1, 160, 160, 3))
    img = np.array(img, dtype=np.float32)
    
    start_ = time.time()
    for i in range(10):
        emb = model.predict(img)
        print(emb[0][:10])
    print(time.time()-start_)