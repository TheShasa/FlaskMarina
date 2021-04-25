
#To run flask
1. export FLASK_APP=app.py
2. export FLASK_DEBUG=1
3. python3 -m flask run

# To create mysql database
1. mysql -h localhost -u root -p(without password)
2. CREATE DATABASE image_embedings;
3. USE image_embedings;
4. CREATE TABLE Embeds(customer_id VARCHAR(20), record_id VARCHAR(20), embed BLOB, from_enrollment BOOL, dt DATETIME);


# Useful mysql functions
* SHOW DATABASES; //show all database
* DROP DATABASE databasename; //delete database
* TRUNCATE TABLE Embeds; //clear the data in table