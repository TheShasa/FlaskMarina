from flask import Flask, request
from flask import jsonify
from flask_mysqldb import MySQL
from utils import raw_to_dict, enroll, verify
import numpy as np
from configparser import ConfigParser
# from ast import literal_eval

config_object = ConfigParser()
config_object.read("config.ini")
app_config = config_object["app"]
app_host = app_config["host"]
app_port = app_config.getint("port")


mysql_config = config_object["mysql"]
mysql_host = mysql_config["host"]
mysql_database = mysql_config["database"]
mysql_user = mysql_config["user"]
mysql_password = mysql_config["password"]

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MYSQL_USER'] = mysql_user
app.config['MYSQL_PASSWORD'] = mysql_password
app.config['MYSQL_HOST'] = mysql_host
app.config['MYSQL_DB'] = mysql_database
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

threshold = 0.5


@app.route('/verification', methods=['POST'])
def verification():

    json_dict = raw_to_dict(request.get_data())
    customer_id = json_dict['customer_id']
    record_id = 0
    image = json_dict['image']
    box = json_dict['box']

    verified, score = verify(image, box, customer_id,
                             record_id, threshold, mysql)
    if verified:
        return jsonify({'status': 200, 'score': score, 'isMatch': True})
    else:
        return jsonify({'status': 200, 'score': score, 'isMatch': False})


@app.route('/enroll', methods=['POST'])
def enrollemnt():
    json_dict = raw_to_dict(request.get_data())
    customer_id = json_dict['customer_id']
    image = json_dict['image']
    box = json_dict['box']

    record_id = 0
    result = {'status': 200, 'id': customer_id}
    # delete old enrollment
    customer_id = json_dict['customer_id']
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM Embeds WHERE customer_id=%s" % customer_id)
    mysql.connection.commit()
    cur.close()

    enroll(image, box, customer_id, record_id, mysql)

    return jsonify(result)


@app.route('/delete_user', methods=['POST'])
def delete_user():
    json_dict = raw_to_dict(request.get_data())
    customer_id = json_dict['customer_id']
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM Embeds WHERE customer_id=%s" % customer_id)
    mysql.connection.commit()
    cur.close()
    return jsonify({'status': 200})


@app.route('/database', methods=['GET'])
def show_db():
    cur = mysql.connection.cursor()
    result = cur.execute("SELECT * FROM Embeds")
    if result > 0:
        data = cur.fetchall()
        res_dict = {}
        for small_dict in data:
            id_key = small_dict['customer_id']

            if id_key in res_dict.keys():
                indict = res_dict[id_key]

                d = {small_dict['record_id']: np.frombuffer(
                    small_dict['embed'], dtype=np.float32).tolist()}
                indict.update(d)
                # res_list[id_key] = indict
            else:
                res_dict[id_key] = {
                    small_dict['record_id']: np.frombuffer(small_dict['embed'],
                                                           dtype=np.float32).tolist()}

        return jsonify(res_dict)
    return jsonify("Database is Empty")


if __name__ == '__main__':
    app.run(debug=True)
