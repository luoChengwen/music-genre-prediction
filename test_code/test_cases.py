# this is a system level test case
from xgboost import XGBClassifier
from functions_for_pred import *

''' due to time limits, only wrote 2 test case'''
'''test case is from raw input to feature extraction and to prediction'''


def test_check_numeric_model_from_input_to_output():
    X_valid = pd.read_csv('./test/X_valid.csv')
    vect_col = pickle.load(open("./model/vect_col.pkl", "rb"))
    num_pred_col = pickle.load(open("./model/numeric_train_col.pkl", "rb"))

    X_valid = create_pca_lda_for_test(X_valid, vect_col)
    X_valid2 = create_new_features(X_valid)

    xgb_model_num = XGBClassifier()
    xgb_model_num.load_model('./model/xgb_model_num.model')
    pred = xgb_model_num.predict(X_valid2[num_pred_col])
    expected_pred = pd.read_csv('./test/valid_test_prediction_num.csv')
    assert sum(expected_pred['pred'] != pred) == 0


def test_check_text_model_from_input_to_output():
    lda_model = gensim.models.ldamulticore.LdaMulticore.load('./model/lda_model.model')
    train_id2word = pickle.load(open("./model/train_id2word.pkl", 'rb'))
    lda_model2 = gensim.models.ldamulticore.LdaMulticore.load('./model/lda_model2.model')
    X_valid = pd.read_csv('./test/X_valid.csv')
    y_valid = pd.read_csv('./test/y_valid.csv')
    X_text_array1, y_text_array1 = prepare_text_for_prediction(X_valid, y_valid, train_id2word, lda_model)
    X_text_array2, y_text_array2 = prepare_text_for_prediction(X_valid, y_valid, train_id2word, lda_model2)
    valid_vecs = np.concatenate([X_text_array1, X_text_array2], axis=1)
    X_valid_vecs = np.array(valid_vecs)

    xgb_model_text = XGBClassifier()
    xgb_model_text.load_model('./model/xgb_model_text.model')
    text_pred = xgb_model_text.predict(X_valid_vecs)

    expected_pred = pd.read_csv('./test/valid_test_prediction.csv')
    assert sum(expected_pred['pred'] != text_pred) < len(text_pred) * .1
