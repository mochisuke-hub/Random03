import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# 学習データの用意
# 学習データの用意
data = {'input_prefecture': [
    '北海道',
    '青森',
    '岩手',
    '宮城件',
    '秋田件',
    '山形剣',
    '福島圏',
    '茨城',
    '栃木',
    '郡馬県',
    'さいたま県',
    '千葉県',
    '東京',
    '神奈川',
    '新潟',
    '富山県',
    '石川',
    '福井',
    '山梨県',
    '長野',
    '岐阜県',
    '静岡県',
    '愛知',
    '三重',
    '滋賀',
    '京都府',
    '大阪',
    '兵庫県',
    '奈良',
    '和歌山',
    '鳥取県',
    '島根',
    '岡山県',
    '広嶋',
    '山口',
    '徳島県',
    '香川県',
    '愛媛県',
    '高知県',
    '福岡県',
    '佐賀県',
    '長崎県',
    '能本県',
    '大分県',
    '宮碕県',
    '鹿児島県',
    '沖縄県'],
        'correct_prefecture': [
    '北海道',
    '青森県',
    '岩手県',
    '宮城県',
    '秋田県',
    '山形県',
    '福島県',
    '茨城県',
    '栃木県',
    '群馬県',
    '埼玉県',
    '千葉県',
    '東京都',
    '神奈川県',
    '新潟県',
    '富山県',
    '石川県',
    '福井県',
    '山梨県',
    '長野県',
    '岐阜県',
    '静岡県',
    '愛知県',
    '三重県',
    '滋賀県',
    '京都府',
    '大阪府',
    '兵庫県',
    '奈良県',
    '和歌山県',
    '鳥取県',
    '島根県',
    '岡山県',
    '広島県',
    '山口県',
    '徳島県',
    '香川県',
    '愛媛県',
    '高知県',
    '福岡県',
    '佐賀県',
    '長崎県',
    '熊本県',
    '大分県',
    '宮崎県',
    '鹿児島県',
    '沖縄県']}

df = pd.DataFrame(data)

# 学習時の処理
training = False  # 学習する場合はTrueに設定
if training:
    print("a")
    # TfidfVectorizerの学習
    # vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4))
    # X_train_tfidf = vectorizer.fit_transform(df['input_prefecture'])
    #
    # # ランダムフォレストモデルの学習
    # model = RandomForestClassifier()
    # model.fit(X_train_tfidf, df['correct_prefecture'])
    #
    # # TfidfVectorizerと学習済みモデルの保存
    # joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    # joblib.dump(model, 'random_forest_model.joblib')
else:
    # 学習済みモデルとTfidfVectorizerの読み込み
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('random_forest_model.joblib')

# 予測
user_input = input('都道府県名を入力してください: ')
user_input_tfidf = vectorizer.transform([user_input])
predicted_prefecture = model.predict(user_input_tfidf)

# 修正された都道府県の表示
print(f'入力した都道府県: {user_input}')
print(f'修正された都道府県: {predicted_prefecture[0]}')