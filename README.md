# Fake-Reviews-Detection

#### About

This Repository contains code that will **Detect a Fake Review** on some online e-commerce site.

We are using a corpus of **Amazon Dataset** present here : [https://www.kaggle.com/lievgarcia/amazon-reviews](https://www.kaggle.com/lievgarcia/amazon-reviews)

##### Requirements

###### 1. Python >= 3.6.1

###### **2. Pip**

---

#### Install virtualenv on your PC

```
pip install virtualenv
```

#### Create a Python Virtual Environment using virtualenv

```
virtualenv venv
```

#### Activate the virtualenv

###### On Linux

```
source venv/bin/activate
```

###### On Windows

```
venv\scripts\activate
```

---

#### Install requirements using pip

```
pip install -r requirements.txt
```

---

#### Train the models

##### There is already a folder called "models" where all the trained models are present.

##### However, if you wanna retrain models, you can run the following command :

```
python main.py --force-retrain 1
```

#### Test the trained models

```
python deploy.py
```

---

#### Run a Flask server for GUI

```
python server.py
```

##### Open 127.0.0.1:5000 in your browser to test your server in action.

---
