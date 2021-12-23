import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():

    def convertSex(word):
        word_dict = {'male':0,'female':1}
        return word_dict[word]

    def convertSmoker(word):
        word_dict = {'no':0,'yes':1}
        return word_dict[word]

    def convertRegion(word):
        word_dict = {'southeast':2,'southwest':3,'northeast':0,'northwest':1}
        return word_dict[word]

    

    age = int(request.form.get('age'))
    sex = convertSex(request.form.get('sex'))
    bmi = float(request.form.get('bmi'))
    print(request.form.get('children'))
    children = int(request.form.get('children'))
    smoker = convertSmoker(request.form.get('smoker'))
    region = convertRegion(request.form.get('region'))

    print(age,sex,bmi,children,smoker,region)

    list=[age,sex,bmi,children,smoker,region]
    final_features=[np.array(list)]
    pred=model.predict(final_features)

    output=pred[0]

    return render_template('index2.html', prediction_text='Premium should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)