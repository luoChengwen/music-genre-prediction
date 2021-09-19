# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020
@author: win10
"""

# 1. Library imports
import uvicorn, base64, io
from fastapi import FastAPI, UploadFile, File, Form
from io import StringIO

# from xgboost import XGBClassifier
from functions_for_pred import *
# import os, gensim, copy, pickle, warnings
# from gensim import corpora, models


# 2. Create the app object
app = FastAPI()

vect_col = pickle.load(open("model/vect_col.pkl", "rb"))
num_pred_col = pickle.load(open("model/numeric_train_col.pkl", "rb"))
lda_model = gensim.models.ldamulticore.LdaMulticore.load('model/lda_model.model')
train_id2word = pickle.load(open("model/train_id2word.pkl", 'rb'))
lda_model2 = gensim.models.ldamulticore.LdaMulticore.load('model/lda_model2.model')
data = pd.read_csv('data/test.csv')
# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     return {'file': pd.read_csv(file.filename)}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence


def combine_title_and_tags(df):
    combine_title_tag_list = []
    for i in range(len(df)):
        df['title'] = df['title'].fillna('')
        df['tags'] = df['tags'].fillna('')
        combine_title_tag_list.append(" ".join([df['title'][i],df['tags'][i]]))
    df = df.assign(text = pd.Series(combine_title_tag_list))
    df = df.drop(columns = ['tags','title'])
    return df


@app.post("/submitform")
async def handle_form(assignment: str = Form(...), assignment_file: UploadFile = File(...)):
    print(assignment_file.filename)
    data = await assignment_file.read()
    s = str(data, 'utf-8')
    data = StringIO(s)
    data = pd.read_csv(data)
    data = combine_title_and_tags(data)
    X_test_num = create_pca_lda_for_test(data, vect_col)
    X_test_num2 = create_new_features(X_test_num)
    X_test_num2 = X_test_num2.drop(columns=['text'])
    print(np.shape(X_test_num2))
    X_test_text_array1, _ = prepare_text_for_prediction(data, np.nan, train_id2word, lda_model)
    X_test_text_array2, _ = prepare_text_for_prediction(data, np.nan, train_id2word, lda_model2)
    final_test_vecs = np.concatenate([X_test_text_array1, X_test_text_array2], axis=1)
    final_feature = np.concatenate([final_test_vecs, np.array(X_test_num2[num_pred_col])], axis=1)
    print(np.shape(final_feature))
    classifier = XGBClassifier()
    classifier.load_model('model/xgb_model_all.model')
    prediction_a = classifier.predict(final_feature)
    print(prediction_a)
    print('--')
    return {'number of records': len(data),
            'number of features': np.shape(data)[1]
        ,'first 10 predictions': list(i.split('\n')[0] for i in prediction_a[:10])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=59698)

# uvicorn app:app --reload

