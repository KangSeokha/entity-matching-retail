import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, classification_report
from fuzzywuzzy import fuzz
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

#File path
path = r"C:\Users\lck\Desktop"

#Load data
train = pd.read_csv(f"{path}/train.csv", encoding='ISO-8859-1')
ltable = pd.read_csv(f"{path}/ltable.csv", encoding='ISO-8859-1')
rtable = pd.read_csv(f"{path}/rtable.csv", encoding='ISO-8859-1')
test = pd.read_csv(f"{path}/test_HW2.csv", encoding='ISO-8859-1')

# Data merge
train_merged = train.merge(ltable, how='left', left_on='ltable_id', right_on='ltable_id', suffixes=('_l', '_r'))
train_merged = train_merged.merge(rtable, how='left', left_on='rtable_id', right_on='rtable_id', suffixes=('_l', '_r'))

test_merged = test.merge(ltable, how='left', left_on='ltable_id', right_on='ltable_id', suffixes=('_l', '_r'))
test_merged = test_merged.merge(rtable, how='left', left_on='rtable_id', right_on='rtable_id', suffixes=('_l', '_r'))

# Missing value handling
def preprocess_data(df):
    df['price_l'] = df['price_l'].fillna(df['price_l'].median())
    df['price_r'] = df['price_r'].fillna(df['price_r'].median())
    df['title_l'] = df['title_l'].fillna('')
    df['title_r'] = df['title_r'].fillna('')
    df['brand_l'] = df['brand_l'].fillna('unknown')
    df['brand_r'] = df['brand_r'].fillna('unknown')
    df['modelno_l'] = df['modelno_l'].fillna('unknown')
    df['modelno_r'] = df['modelno_r'].fillna('unknown')
    return df

train_merged = preprocess_data(train_merged)
test_merged = preprocess_data(test_merged)

# Feature engineering
def create_features(df):
    features = pd.DataFrame()
    
    # Text similarity (title)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['title_l'] + " " + df['brand_l'])
    tfidf_matrix_r = tfidf.transform(df['title_r'] + " " + df['brand_r'])
    features['title_cosine_sim'] = [cosine_similarity(tfidf_matrix[i], tfidf_matrix_r[i])[0, 0] for i in range(len(df))]
    
   # Fuzzy matching features
    features['brand_fuzzy_ratio'] = df.apply(lambda row: fuzz.ratio(row['brand_l'], row['brand_r']), axis=1)
    features['model_fuzzy_ratio'] = df.apply(lambda row: fuzz.ratio(row['modelno_l'], row['modelno_r']), axis=1)
    
    # Price difference
    features['price_diff'] = np.abs(df['price_l'] - df['price_r'])
    features['price_ratio'] = df['price_l'] / (df['price_r'] + 1e-5)
    
    #brandmatch
    features['brand_match'] = (df['brand_l'] == df['brand_r']).astype(int)
    
    # Model matching
    features['modelno_match'] = (df['modelno_l'] == df['modelno_r']).astype(int)
    
    # Text length difference
    features['title_len_diff'] = np.abs(df['title_l'].str.len() - df['title_r'].str.len())
    
    # 标题关键词匹配
    def keyword_overlap(row):
        l_keywords = set(row['title_l'].split())
        r_keywords = set(row['title_r'].split())
        if len(l_keywords | r_keywords) == 0:
            return 0
        return len(l_keywords & r_keywords) / len(l_keywords | r_keywords)
    
    features['title_keyword_overlap'] = df.apply(keyword_overlap, axis=1)
    
    return features


X_train = create_features(train_merged)
y_train = train['label']

X_test = create_features(test_merged)

# Divide training set and validation set
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Random forest model - adjust category weights
rf_model = RandomForestClassifier(
    n_estimators=300, 
    max_depth=None, 
    min_samples_leaf=3, 
    min_samples_split=2, 
    class_weight={0: 1, 1: 5},  # Increase the weight of category 1
    random_state=42
)
rf_model.fit(X_tr, y_tr)

#LightGBM model
lgb_model = LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',
    learning_rate=0.1,
    random_state=42
)
lgb_model.fit(X_tr, y_tr)

# Integrated model
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model), 
        ('lgb', lgb_model)
    ], 
    voting='soft'
)
ensemble_model.fit(X_tr, y_tr)

# Validation set evaluation
y_pred_ensemble = ensemble_model.predict(X_val)
print("# Integrated model F1-Score:", f1_score(y_val, y_pred_ensemble))
print("\nClassification report:\n", classification_report(y_val, y_pred_ensemble))

# Test set prediction
test_preds = ensemble_model.predict(X_test)

# Feature importance visualization (random forest model)
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title("Feature Importance - Random Forest")
plt.show()

# Save submission file
submission = pd.DataFrame({'id': test['id'], 'label': test_preds})
submission.to_csv(f"{path}/submission.csv", index=False)
print("Submission saved to:", f"{path}/submission.csv")
