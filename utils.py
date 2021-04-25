
import numpy as np
import pandas as pd

from ast import literal_eval
import time
from io import BytesIO
import base64

from PIL import Image
from keras_facenet import FaceNet
import cv2
encoder = FaceNet().model


def preprocessing(self, img):
    image_size = 160
    img = cv2.resize(img, (image_size, image_size))
    img = np.expand_dims(np.array(img), axis=0)
    return (np.float32(img) - 127.5) / 127.5


def get_embedding(image):
    image = preprocessing(image)
    # x = np.random.rand(256)
    # x = x / np.sqrt(np.sum(x ** 2))
    # return x
    return encoder.predict(image)


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
    image = image[box[0]:box[0]+box[2], box[1]:box[1]+box[3]]
    embed = get_embedding(image)

    print(embed)
    if embed is None:
        return False
    bf = embed.tobytes()  # np.getbuffer(embed, dtype=np.float32)
    cur = mysql.connection.cursor()

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
    embeds = np.zeros((result, 256))
    # get teh data
    if result > 0 and current_embed is not None:

        # get the data from select and save to arrays
        data = cur.fetchall()
        for i, _dict in enumerate(data):
            ids[i] = int(_dict['customer_id'])
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

    current_embed = base64_to_image(image)
    image = image[box[0]:box[0]+box[2], box[1]:box[1]+box[3]]
    cur = mysql.connection.cursor()
    result = cur.execute(
        "SELECT * FROM Embeds WHERE customer_id=(%s)", [customer_id])
    ids = np.zeros(result, dtype=np.int32)
    embeds = np.zeros((result, 256))
    # get the data
    if result > 0 and current_embed is not None:

        # get the data from select and save to arrays
        data = cur.fetchall()
        for i, _dict in enumerate(data):
            ids[i] = int(_dict['customer_id'])
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
