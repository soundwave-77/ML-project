import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


# function for preparing text data
# def prepare_text_data(df, text_col):
#     stopWords = set(stopwords.words('russian'))
#     vectorizer = TfidfVectorizer(stop_words=list(stopWords), max_features=2000)
#     vectorizer.fit(df[text_col])
#     return vectorizer
    # train_title = vectorizer.transform(df_train['title'])
    # test_title = vectorizer.transform(df_test['title'])
    # train_title_df = pd.DataFrame.sparse.from_spmatrix(train_title, columns=vectorizer.get_feature_names_out())
    # test_title_df = pd.DataFrame.sparse.from_spmatrix(test_title, columns=vectorizer.get_feature_names_out())


def prepare_price_col(df):
    """
    Function to fill missing values in price column with mean price
    """
    df['price'] = df['price'].fillna(df['price'].mean())
    return df


def fill_missing_values(df, cols):
    for col in cols:
        df[col] = df[col].fillna('')
    return df


def load_and_preprocess_data(train_path, test_path):
    print("==== Preparing Data ====")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)


    # prepare price column
    df_train = prepare_price_col(df_train)
    df_test = prepare_price_col(df_test)


    # fill missing values
    col_list = ['param_1', 'param_2', 'param_3', 'image_top_1', 'title', 'description']
    df_train = fill_missing_values(df_train, col_list)
    df_test = fill_missing_values(df_test, col_list)


    # convert image_top_1 to string
    df_train['image_top_1'] = df_train['image_top_1'].astype('str')
    df_test['image_top_1'] = df_test['image_top_1'].astype('str')


    # drop unused(for now) columns
    drop_cols = ['image', 'item_id', 'user_id', 'activation_date', 'deal_prob_cat', 'params', 'price_log', 'title', 'description']
    df_train.drop(drop_cols, axis=1, inplace=True)
    df_test.drop(drop_cols, axis=1, inplace=True)

    X = df_train.drop(columns=['deal_probability'])
    y = df_train['deal_probability']

    print("==== Data preprocessed successfully! ====")

    cat_features = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1']

    return X, y, cat_features


def main(train_path = 'data/raw/train.csv', test_path = 'data/raw/test.csv'):
    """
    Script to prepare data for training
    
    Receives raw data and converts it into csv file for training
    """
    # Load and preprocess data
    X, y, cat_features = load_and_preprocess_data(train_path, test_path)

    # # Save data to csv
    # X.to_csv('data/prepared/X_train.csv', index=False)
    # y.to_csv('data/prepared/y_train.csv', index=False)
    # cat_features.to_csv('data/prepared/cat_features.csv', index=False)


if __name__ == '__main__':
    main()
