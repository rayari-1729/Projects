from flask import Flask, render_template, request  
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model


# from flask_ngrok import run_with_ngrok

app = Flask(__name__)
# run_with_ngrok(app)

# Load the Tokenizer file
with open(r"tokenizer.pkl", "rb") as f:
    token = pickle.load(f)

# Load the lstm model
model = load_model('model.h5')

# Render the HTML file for the home page
@app.route("/",methods=['GET','POST'])
def home():
    result = {}
    if request.method == 'POST':
        message = request.form['message']
        result = predict(message)
        return render_template('index.html', result=result)
    return render_template('index.html', result=result)

def predict(message):
    
    # Take a string input from user
    user_input = message
    data = [user_input]

    tokenize_data = token.texts_to_sequences(data)
    final_data = pad_sequences(tokenize_data, maxlen=200)
    ans_prob = model.predict(final_data)

    pred_tox = ans_prob[0][0]*100
    pred_sev = ans_prob[0][1]*100
    pred_obs = ans_prob[0][2]*100
    pred_thr = ans_prob[0][3]*100
    pred_ins = ans_prob[0][4]*100
    pred_ide = ans_prob[0][5]*100

    out_tox = round(pred_tox, 2)
    out_sev = round(pred_sev, 2)
    out_obs = round(pred_obs, 2)
    out_ins = round(pred_ins, 2)
    out_thr = round(pred_thr, 2)
    out_ide = round(pred_ide, 2)

    print(out_tox)

    result ={}

    result['Probability of being a Toxic comment: '] = out_tox
    result['Probability of being a Severely toxic comment: '] = out_sev
    result['Probability of being an Obscene comment: '] = out_obs
    result['Probability of being an Insult: '] =  out_ins
    result['Probability of being a Threat: '] = out_thr
    result['Probability of being an Identity Hate comment: '] = out_ide                        
    
    return result
     
# Server reloads itself if code changes so no need to keep restarting:
if __name__ == "__main__":
    app.run()