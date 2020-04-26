import flask
import pickle
import joblib
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Use pickle to load in the pre-trained model.
#model = pickle.load(open(f'model/model_xgb.pkl','rb'))
with open('model/model_rf.pkl', 'rb') as f:
   model = pickle.load(f)

#xgb = joblib.load('model/model.pkl.z')

#loaded_model = pickle.load(open("model/pima.pickle.dat", "rb"))

dict1 = pickle.load(open('model/dict.pkl','rb'))

def preprocess(content):
    result = []
    for s in content:
        data = word_tokenize(s.lower())
        stop_words = set(stopwords.words('english'))
        data = [w for w in data if not w in stop_words]
        lemmatizer = WordNetLemmatizer()
        data_lem = [lemmatizer.lemmatize(w) for w in data]
        data_lem = sorted(list(set(data_lem)))

        data = ' '.join(word for word in data_lem)
        result.append(data)
    return result

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        url = flask.request.form['URL']
        client_id = ''
        client_secret =  ''
        user_agent = ''
        username = ''
        password = ''

        import praw

        reddit = praw.Reddit(client_id = client_id, client_secret = client_secret, user_agent = user_agent, username = username, password = password)

        id = url.split("/")[-3]
        sub = reddit.submission(id)
        title = sub.title

        clf = TfidfVectorizer(input = 'content', vocabulary = dict1)
        X = clf.fit_transform(preprocess([title]))
        clf.get_feature_names()
        X = X.toarray()
        print(X.shape)

        prediction = model.predict(X)
        f = {0:'Non-Political', 1:'Coronavirus', 2:'AskIndia', 3:'Photography', 4:'Science/Technology', 5:'Politics', 6:'Policy/Economy', 7:'Business/Finance', 8:'Sports', 9:'Food'}
        pred = f.get(prediction[0])

        return flask.render_template('main.html',
                                     original_input={'URL':url,
                                                     },
                                     result=pred,
                                     )


if __name__ == '__main__':
    app.run()
