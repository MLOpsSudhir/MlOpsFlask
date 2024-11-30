import os
import pickle
from flask import Flask,render_template,request

model_path = os.path.abspath("artifacts/model_evaluation")
model_path_files_name = os.path.join(model_path,"perfect_model.pkl")
model = pickle.load(open(model_path_files_name,'rb'))

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])

def prediction():
    if request.method == "POST":
        data_dict = dict(request.form)
        num1 = float(data_dict["First_number"])
        prediction = model.predict([[num1]])[0][0]
        return render_template("index.html",response=prediction)
    else:
        return render_template("index.html")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)
