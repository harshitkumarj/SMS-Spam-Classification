from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import joblib

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    import nltk
    import sklearn
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    df = pd.read_table('SMSSPamCollection', sep='\t',header=None, encoding='utf-8',names=['label','message'])
    stemmer = PorterStemmer()
    corpus=[]
    for i in range(len(df)):
        review = re.sub('[^a-zA-Z]',' ',df['message'][i])
        review = review.lower()
        review = review.split()
    
        review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
	
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer()
    X = tf.fit_transform(corpus).toarray()
    
    message = request.form['message']
    data = [message]
    vect = tf.transform(data).toarray()
    my_prediction = model.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)