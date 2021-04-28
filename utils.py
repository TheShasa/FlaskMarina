
import numpy as np
import pandas as pd

from ast import literal_eval
import time
from io import BytesIO
import base64
from PIL import Image
import cv2
from model_arch import IrisModel
from datetime import datetime_CAPI
import os
from tflit_model import TensorFlowModel
import mtcnn
from datetime import datetime

face_detector = mtcnn.MTCNN()
encoder = TensorFlowModel()
encoder.load(os.path.join(os.getcwd(), 'facenet_tflite/model.tflite'))


def preprocessing(img):
    image_size = 160
    img = cv2.resize(img, (image_size, image_size))
    img = np.expand_dims(np.array(img), axis=0)
    return (np.float32(img) - 127.5) / 127.5


def get_embedding(image):
    image = preprocessing(image)
    image = np.array(image, dtype=np.float32)
    # x = np.random.rand(256)
    # x = x / np.sqrt(np.sum(x ** 2))
    # return x
    return encoder.predict(image)


def crop(image, box):
    h, w = image.shape[:2]
    pred = face_detector.detect_faces(image)
    # print(h, w)
    # lr, rr = get_left_right_ear(image)

    if len(pred) == 0:  # did not found face
        left = box[0]
        right = box[0]+box[2]
    else:  # use box from react
        left_eye = pred[0]['keypoints']['left_eye']
        right_eye = pred[0]['keypoints']['right_eye']
        left = min(left_eye[0], right_eye[0])
        right = max(left_eye[0], right_eye[0])
        width = abs(right-left)
        left -= width//2
        right += width//2
    return image[
        max(0, box[1]):min(h, box[1]+box[3]),
        #  max(0, box[0]):min(h, box[0]+box[2])]
        max(0, left):min(w, right)]


def base64_to_image(img_base64):
    img = Image.open(BytesIO(base64.b64decode(img_base64)))
    img = np.array(img, dtype='uint8')
    return img


def image_to_base64(image):
    pil_img = Image.fromarray(image)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return str(new_image_string)


def enroll(image, box, customer_id, record_id, mysql, from_enroll=1):
    image = base64_to_image(image)

    image = cv2.resize(image, (box[5], box[4]))
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    print('storage/'+str(customer_id)+"_"+date_time+"_"+"_1.png")
    cv2.imwrite('storage/'+str(customer_id)+"_"+date_time+"_"+"_1.png", image)

    image = crop(image, box)
    cv2.imwrite('storage/'+str(customer_id)+"_"+date_time+"_"+'_2.png', image)

    embed = get_embedding(image)

    if embed is None:
        return False
    bf = embed.tobytes()  # np.getbuffer(embed, dtype=np.float32)
    cur = mysql.connection.cursor()
    print(customer_id)
    cur.execute("INSERT INTO Embeds(customer_id,record_id, embed, from_enrollment, dt) VALUES(%s, %s, %s, %s, %s)",
                (customer_id, record_id, bf, from_enroll, time.strftime('%Y-%m-%d %H:%M:%S')))
    mysql.connection.commit()
    cur.close()
    return True


def recognize(image, threshold, mysql):
    image = base64_to_image(image)
    current_embed = get_embedding(image)
    if current_embed is None:
        return None
    cur = mysql.connection.cursor()
    result = cur.execute("SELECT * FROM Embeds")
    ids = np.zeros(result, dtype=np.int32)
    embeds = np.zeros((result, 512))
    # get teh data
    if result > 0 and current_embed is not None:

        # get the data from select and save to arrays
        data = cur.fetchall()
        for i, _dict in enumerate(data):
            ids[i] = _dict['customer_id']
            embeds[i] = np.frombuffer(_dict['embed'], dtype=np.float32)

        # calculate the score with current_embed
        utt_sim_matrix = np.inner(current_embed, embeds)

        df = pd.DataFrame()
        df['id'] = ids
        df['score'] = utt_sim_matrix
        df = df[df['score'] > threshold]
        df.reset_index(inplace=True, drop=True)

        df = df.groupby(by=['id']).max()
        df.sort_values(by=['score'], ascending=False, inplace=True)

        # best id by score
        best_id = df.index.values
        best_id = best_id[:min(len(best_id), 5)]

        # result dict for jsonify
        res_dict = []
        for id in best_id:
            res_dict.append({'score': df.loc[id].values[0],
                             'id': int(id)})
        return res_dict

    return None


def verify(image, box, customer_id, record_id, threshold, mysql):

    image = np.random.random((224,224,3))#base64_to_image(image)
    image = cv2.resize(image, (box[5], box[4]))
    image = crop(image, box)
    current_embed = get_embedding(image)

    cur = mysql.connection.cursor()
    result = cur.execute(
        "SELECT * FROM Embeds WHERE customer_id=(%s)", [customer_id])
    embeds = np.zeros((result, 512))
    # get the data
    if result > 0 and current_embed is not None:

        # get the data from select and save to arrays
        data = cur.fetchall()
        for i, _dict in enumerate(data):
            embeds[i] = np.frombuffer(_dict['embed'], dtype=np.float32)
        # calculate the score with current_embed
        utt_sim_matrix = np.inner(current_embed, embeds)
        max_value = utt_sim_matrix.max()

        # if max_value > 0.5:
        #     enroll(image, box, customer_id, record_id, mysql, 0)

        if max_value > threshold:
            return True, max_value
        else:
            return False, max_value
    return False, 0


def raw_to_dict(bytes_dict):
    dict_str = bytes_dict.decode("UTF-8")
    return literal_eval(dict_str)
