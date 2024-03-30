import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

app = Flask(__name__)  # initializing the flask application
model = load_model(r'C:\Users\kalpa\Downloads\Flask\model.h5')  # loading the model

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x))
        index =['Diabetic Retinopathy','Gluacoma','Cataract','Normal']
       
        text="The Classified image is : "+ str(index[pred])
    return text

if __name__ == '__main__':
    app.run(debug=True)  # run the flask application
