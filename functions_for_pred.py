import numpy as np
import pandas as pd # ( pd.read_csv)
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score
import os, gensim, copy, pickle, warnings
from gensim.utils import simple_preprocess
from sklearn.feature_selection import SelectFromModel
from nltk.corpus import stopwords
# import numpy as np
from gensim import corpora, models
from sklearn.model_selection import GridSearchCV
import seaborn as sns
np.random.seed(400)
from imblearn.over_sampling import SMOTE
import pyLDAvis
import pyLDAvis.gensim_models
import nltk, pickle
from sklearn.decomposition import PCA
import pickle
nltk.download('wordnet')
nltk.download('stopwords')


def apply_pca(X, comp, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Create principal components
    pca = PCA(n_components=comp)
    X_pca = pca.fit_transform(X.fillna(0))

    # Convert to dataframe
    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=X.index)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    pickle.dump(pca, open("model/pca.pkl", "wb"))

    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # plot variance of PCA and determine number of cluster
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


def perform_lda(X, y, vect_col):
    # perform LDA
    X2 = X[vect_col]
    X2 = X2.fillna(0)
    X2 = (X2 - X2.mean(axis=0)) / X2.std(axis=0)
    lda = LinearDiscriminantAnalysis(n_components=len(set(y)) - 1)
    X_lda = lda.fit(X2, y)
    pickle.dump(X_lda, open("model/lda.pkl", "wb"))
    X_lda = pd.DataFrame(X_lda.transform(X[vect_col].fillna(0)), index=X2.index)
    X_lda.columns = ['LA' + str(i + 1) for i in range(np.shape(X_lda)[1])]
    assert (len(X) == len(X_lda))
    return X.merge(X_lda, left_index=True, right_index=True)


def merge_dfs(df1, df_pca, df_lda, col_to_drop):
    df1 = df1.merge(df_pca, left_index=True, right_index=True)
    df1 = df1.merge(df_lda, left_index=True, right_index=True)
    return df1  # .drop(columns = col_to_drop)


def create_new_features(df):
    # since a few variables have limited set - 15 time_signature, 27 key, 11 mode
    # can randomly combine these variables to see if it helps
    df = df.fillna(0)
    df['mode_key'] = (df['mode'] ** 2 + 10) * (df['key'] ** 2)
    df['key_time_sig'] = (df['time_signature'] ** 2 + 10) * df['key']
    df['mode_time_sig'] = (df['mode'] ** 2 + 1) * (df['time_signature'] ** 2 + 100)
    df['key_mode_time_sig'] = (df['mode'] ** 2 + 5) * (df['key'] + 1) * (df['time_signature'] + 1)
    df['mode_loudness'] = (df['mode'] ** 2 + 5) * (df['loudness'] + 10) ** 2
    df['temp_mode'] = (df['mode'] + 5) * df['tempo'] ** 2
    df['temp_mode_key'] = (df['mode'] + 5) ** 2 * (df['tempo'] + 10) ** 2 * (df['key'] + 10) ** 2
    return df


def smote2(X, y):
    X1, y1 = copy.deepcopy(X), copy.deepcopy(y)  # init
    sm = SMOTE(random_state=2)
    X1, y1 = sm.fit_resample(X, y)
    return X1, y1


def create_pca_lda_for_test(df_test, vect_col):
    pca = pickle.load(open("model/pca.pkl", 'rb'))
    lda = pickle.load(open("model/lda.pkl", 'rb'))

    df_test_pca = pd.DataFrame(pca.transform(df_test[vect_col].fillna(
        df_test[vect_col].median())), columns=['PC1', 'PC2', 'PC3'], index=df_test.index)
    df_test_lda = pd.DataFrame(lda.transform(df_test[vect_col].fillna(0)), index=df_test.index)
    df_test_lda.columns = ['LA' + str(i + 1) for i in range(np.shape(df_test_lda)[1])]
    df_test = df_test.merge(df_test_lda, left_index=True, right_index=True)
    df_test = df_test.merge(df_test_pca, left_index=True, right_index=True)
    assert len(df_test) == len(df_test_pca) == len(df_test_lda)
    return df_test


def strip_newline(series):
    return [review.replace('\n', '') for review in series]


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['let', 'oh', 'got', 'hey', 'hay', 'ya', 'ooh', 'go', 'ai',
                       'ca', 'na', 'say', 'sure', 'yeah', 'tu', 'els', 'might', 'done'])  # 'may','day'
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_corpus(df):
    df['text'] = strip_newline(df.text)
    words = list(sent_to_words(df.text))
    words = remove_stopwords(words)
    bigram_mod = bigrams(words)
    bigram = [bigram_mod[review] for review in words]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=12, no_above=0.5)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram


## create vectors
def get_text_vector(df, corpuss, lda_model):
    vecs = []
    for i in range(len(df)):
        top_topics = lda_model.get_document_topics(corpuss[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(8)]
        topic_vec.extend([len(df.iloc[i].text)])  # length review
        vecs.append(topic_vec)
    return vecs


## pre-processing by split the tag and title to list
def combine_title_and_tags(df):
    combine_title_tag_list = []
    for i in range(len(df)):
        df['title'] = df['title'].fillna('')
        df['tags'] = df['tags'].fillna('')
        combine_title_tag_list.append(" ".join([df['title'][i],df['tags'][i]]))
    df = df.assign(text = pd.Series(combine_title_tag_list))
    df = df.drop(columns = ['tags','title'])
    return df


def get_bigram(df):
    df['text'] = strip_newline(df.text)
    words = list(sent_to_words(df.text))
    words = remove_stopwords(words)
    bigram = bigrams(words)
    bigram = [bigram[i] for i in words]
    return bigram


def prepare_text_for_prediction(X_text_df, y_text_df, id2word_dict,lda_model):
    bigram_df = get_bigram(X_text_df)
    df_corpus = [id2word_dict.doc2bow(text) for text in bigram_df]
    df_vecs = get_text_vector(X_text_df, df_corpus, lda_model)
    X_text_array = np.array(df_vecs)
    y_text_array = np.array(y_text_df)
    return X_text_array, y_text_array


def grid_search_CV(grid_para, X, y, test_X, test_y, name):
    model_1 = GridSearchCV(XGBClassifier(eval_metric='mlogloss'),
                           grid_para, cv=3)

    pickle.dump(X.columns, open("model/numeric_train_col.pkl", "wb"))

    X = np.array(X)
    y = np.array(y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    model_1.fit(X, y, eval_set=[(test_X, test_y)],
                early_stopping_rounds=10)
    print(model_1.best_score_)
    print(model_1.best_params_)

    xgb_m = XGBClassifier(eval_metric=['mlogloss'],
                          objective='multi:softprob',
                          booster='gbtree',
                          grow_policy='lossguide',
                          max_depth=model_1.best_params_['max_depth'],
                          reg_alpha=model_1.best_params_['reg_alpha'],
                          reg_lambda=model_1.best_params_['reg_lambda'],
                          eta=model_1.best_params_['eta'],
                          num_parallel_tree=model_1.best_params_['num_parallel_tree'],
                          gamma=model_1.best_params_['gamma'],
                          sampling_method=model_1.best_params_['sampling_method'],
                          colsample_bytree=model_1.best_params_['colsample_bytree'],
                          seed=123,
                          n_jobs=-1,
                          )
    xgb_m.fit(X, y)
    xgb_m.save_model('model/xgb_model_' + name + '_.model')

    return xgb_m
