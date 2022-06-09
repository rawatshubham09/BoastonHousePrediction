from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle


app = Flask(__name__)

knn = pickle.load(open("Knn.pkl", "rb"))
lr = pickle.load(open("LinearModel.pkl", "rb"))
st = pickle.load(open("Standard_scaler.pkl", "rb"))

@app.route('/', methods=['GET', 'POST']) 
def home():
    return render_template('home.html')

def process_data(data):
    data1 = data.copy()
    for i, val in enumerate(data1):
        if i in [0, 2, 5, 7]:
            if data[i] > 0:
                data1[i] = np.log10(data1[i]+0.000001)
        if i in [4, 6]:
            data1[i] = data1[i]**2
    return data1

@app.route('/predict', methods=['POST']) 
def form_page():
    try:
        if (request.method == 'POST'):
            CRIM = float(request.form.get("CRIM"))
            INDUS = float(request.form.get("INDUS"))
            NOX = float(request.form.get("NOX"))
            RM = float(request.form.get("RM"))
            AGE = float(request.form.get("AGE"))
            TAX = float(request.form.get("TAX"))
            PTRATIO = float(request.form.get("PTRATIO"))
            LSTAT = float(request.form.get("LSTAT"))

        val = list((CRIM, INDUS, NOX, RM, AGE, TAX, PTRATIO, LSTAT))
        print("data Collected : ", val)

        # Processing Right and Left Skew Data
        testData = [process_data(val)]

        # Predicting Group
        grp = knn.predict(testData)

        # Standard Scaler
        testData = st.transform(testData)

        # Hstacking both data
        testData = np.hstack((testData, [grp]))

        # Predicting Price
        price = lr.predict(testData)[0]
        price = round(price, 2)

        return render_template("result.html", price=price)
    except:
        print("Form Page problem")
        return render_template('home.html')
    
@app.route('/predict_api', methods=["GET","POST"])
def test():
    try:
        if request.method == "POST":
            data_json = request.json["data"]
            print(data_json)
            CRIM = data_json["CRIM"]
            INDUS = data_json["INDUS"]
            NOX = data_json["NOX"]
            RM = data_json["RM"]
            AGE = data_json["AGE"]
            TAX = data_json["TAX"]
            PTRATIO = data_json["PTRATIO"]
            LSTAT = data_json["LSTAT"]

            val = list((CRIM, INDUS, NOX, RM, AGE, TAX, PTRATIO, LSTAT))
            print("data Collected : ", val)

            # Processing Right and Left Skew Data
            testData = [process_data(val)]
            print("After Process : ", testData)
            # Predicting Group
            grp = knn.predict(testData)
            print("Group no. : ", grp)
            # Standard Scaler
            testData = st.transform(testData)
            print("After Transformation : ", testData)
            # Hstacking both data
            testData = np.hstack((testData, [grp]))
            print("HStack data : ", testData)

            # Predicting Price
            price = lr.predict(testData)
            print("Price : ", price[0])

            price = round(price[0], 2)
            return jsonify({"Price" : price})
    except :
        return jsonify({"Responce" : "Bad Request"})

if __name__=="__main__":
    app.run(debug=True)