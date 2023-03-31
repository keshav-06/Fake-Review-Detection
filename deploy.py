import os
import re
import pickle, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# nltk.download()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')


categories = ["Apparel", "Automotive", "Baby", "Beauty", "Books", "Camera", "Electronics", "Furniture", "Grocery", "Health & Personal Care", "Home", "Home Entertainment", "Home Improvement", "Jewelry", "Kitchen", "Lawn and Garden", "Luggage", "Musical Instruments", "Office Products", "Outdoors", "PC", "Pet Products", "Shoes", "Sports", "Tools", "Toys", "Video DVD", "Video Games", "Watches", "Wireless"]
categories_str = "Apparel"
for i in range (1, len(categories)) :
    categories_str += ", " + categories[i]

def clean_review(review):
    review = re.sub('[^a-zA-Z]',' ', review)
    review = review.lower()
    review = review.split()
    #print (review)
    review = [word for word in review if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

def countvectorize(statement):
    countvectorizer = pickle.load(open(os.path.join("models", "countvectorizer.sav"), 'rb'))
    statement = countvectorizer.transform(statement).toarray()
    return statement


def onehotencode(rating, verified_purchase, product_category, X):
    labelencoder_1 = pickle.load(open(os.path.join("models", "labelencoder_1.sav"), 'rb'))
    labelencoder_2 = pickle.load(open(os.path.join("models", "labelencoder_2.sav"), 'rb'))
    labelencoder_3 = pickle.load(open(os.path.join("models", "labelencoder_3.sav"), 'rb'))

    ct1 = pickle.load(open(os.path.join("models", "columntransformer1.sav"), 'rb'))
    ct2 = pickle.load(open(os.path.join("models", "columntransformer2.sav"), 'rb'))
    ct3 = pickle.load(open(os.path.join("models", "columntransformer3.sav"), 'rb'))

    w, h = 3, 1;
    new_col = [[0 for x in range(w)] for y in range(h)]
    num = 0

    for i in range(0, 1):
        new_col[i][0] = rating
        new_col[i][1] = verified_purchase
        new_col[i][2] = product_category

    new_col = np.array(new_col)

    new_col[:, 0] = labelencoder_1.transform(new_col[:, 0])
    new_col[:, 1] = labelencoder_2.transform(new_col[:, 1])
    new_col[:, 2] = labelencoder_3.transform(new_col[:, 2])

    new_col = ct1.transform(new_col)
    try:
        new_col = new_col.toarray()
    except:
        #Do Nothing
        pass
    new_col = new_col.astype(np.float64)

    new_col = ct2.transform(new_col)
    try:
        new_col = new_col.toarray()
    except:
        #Do Nothing
        pass
    new_col = new_col.astype(np.float64)

    new_col = ct3.transform(new_col)
    try:
        new_col = new_col.toarray()
    except:
        #Do Nothing
        pass
    new_col = new_col.astype(np.float64)

    X= np.append(X, new_col, axis=1)
    return X

def POS_Tagging(sentence):
    tagged_list = []
    tags = []
    count_verbs = 0
    count_nouns = 0
    text=nltk.word_tokenize(sentence)
    tagged_list = (nltk.pos_tag(text))

    tags = [x[1] for x in tagged_list]
    for each_item in tags:
        if each_item in ['VERB','VB','VBN','VBD','VBZ','VBG','VBP']:
            count_verbs+=1
        elif each_item in ['NOUN','NNP','NN','NUM','NNS','NP','NNPS']:
            count_nouns+=1
        else:
            continue
    if count_verbs > count_nouns:
        sentence = 'F'
    else:
        sentence = 'T'

    return sentence


def postag(sentence, X):
    w, h = 2, 1;
    pos_tag = [[0 for x in range(w)] for y in range(h)]
    num = 0

    sentence = POS_Tagging(sentence)

    if sentence=='T':
        pos_tag[0][0] = 1
        pos_tag[0][1] = 0
    else:
        pos_tag[0][0] = 0
        pos_tag[0][1] = 1

    X = np.append(X, pos_tag, axis=1)
    return X


def classify(X):
    bernoullinb = pickle.load(open(os.path.join("models", "bernoullinb.sav"), 'rb'))
    return bernoullinb.predict(X)

def get_result(statement, rating, verified_purchase, product_category):
    X = countvectorize([statement])
    X = postag(statement, X)
    X = onehotencode(rating, verified_purchase, product_category, X)

    X = classify(X)
    return X

def test_input(product_rating, verified_purchase, product_category) :
    x = True
    y = True
    z = True

    if product_rating != '1' and product_rating != '2' and product_rating != '3' and product_rating != '4' and product_rating != '5' :
        print ("--------------------------------------------------------------------------------------.")
        print ("\nError : Product Rating must be Between 1 and 5 (inclusive).")
        print ("\nPlease Try Again.")

        x = False

    if verified_purchase != 'Y' and verified_purchase != 'N' :
        print ("--------------------------------------------------------------------------------------.")
        print ("\nError : Verified Purchase must be either Y (Yes) or N (No).")
        print ("\nPlease Try Again.")

        y = False

    if product_category not in categories:
        print ("--------------------------------------------------------------------------------------.")
        print ("\nError : Product Category must be among these choices : \n" + categories_str)
        print ("\nPlease Try Again.")

        z = False

    return [x, y, z]

if __name__ == '__main__':

    review_text = input("\nEnter your Review : ")

    product_rating = ""
    verified_purchase = ""
    product_category = ""

    input_ar = [False, False, False]

    while (True) :
        print("\n---------------------------------------------------------------------------------------\n")

        if not input_ar[0] :
            product_rating = input("\nEnter your Product Rating (On a scale of 1 to 5) : ")

        if not input_ar[1] :
            verified_purchase = input("\nEnter if it's a Verified Purchase (Y or N) : ")

        if not input_ar[2] :
            product_category = input("\nEnter your Product Category (" + categories_str + ") : ")

        input_ar = test_input(product_rating, verified_purchase, product_category)

        if input_ar == [True, True, True] :
            break

    answer = get_result(review_text, product_rating, verified_purchase, product_category)

    if answer == 1:
        print ("It is a True Review")

    else:
        print ("It is a False Review")
