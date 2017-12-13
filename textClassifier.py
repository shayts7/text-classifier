#TextClassifier gives the ability to predict a label based on a free text (the feature)
class TextClassifier:
    
    # init the classifier with array of json objects to train the model
    # feature_name - the json field name that contains the free text to predict according
    # label_name - the json field name that contains the label (the prediction target)
    def __init__(self, to_train_json, feature_name, label_name):
        
        # read json into pandas
        import pandas as pd
        train_df = pd.read_json(to_train_json, orient='records')
        self.feature_name = feature_name
        self.label_name = label_name
        
        # remove non-alphabetic and unicode chars
        train_df.description.replace({r'\W+':' '}, regex=True, inplace=True)
        train_df[feature_name].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

        # train the classifier
        text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                    ('tfidf', TfidfTransformer()),
                    ('clf-svm', SGDClassifier(loss='modified_huber',penalty='l2',alpha=0.003, n_iter=5, random_state=30))])
        self.text_clf_svm = text_clf_svm.fit(train[feature_name].values, train[label_name].values)
     
    # gets array of json object to predict (accoring the feature_name initalized in the ctor)
    # returns the same array with additinal json property - the predicted label (property name set to label_name)
    def getBatchSegment(self, to_predict_json):
        
        # predict and set back to the data frame
        to_predict_df = pd.read_json(to_predict_json, orient='records')
        predicted = self.text_clf_svm.predict(to_predict_df[self.feature_name].values)
        to_predict_df[self.label_name] = predicted
        
        # in case of non english feature - we set the label "Not supported language"
        # please notice - this is a performance sensative operation
        from langdetect import detect
        for index, row in to_predict_df.iterrows():
            if detect(row[self.feature_name]) != 'en':
                to_predict_df.set_value(index,self.label_name,'Not supported language')
            
        return to_predict_df.to_json(orient='records')
    
    
    # get the most suitable labels for the given json object
    # top_count - number of labels to return
    def getTopSegments(self, to_predict_json, top_count):
        
        to_predict_df = pd.read_json(to_predict_json, orient='records')
        feature_to_predict = to_predict_df[self.feature_name].iloc[0]
        
        #zip the probs and labels, sorting by probs and returning top labels
        data = zip(self.text_clf_svm.predict_proba(to_predict_df[self.feature_name].values)[0], self.text_clf_svm.classes_)
        data.sort(key=lambda tup: tup[0], reverse=True) 
        unzipped = zip(*data)
        labels = unzipped[1][0:top_count]
        return labels
    
    # returns labels set
    def getSegments(self):
        return self.text_clf_svm.classes_
        
     
# usage example
import pandas as pd
train_df = pd.read_excel('appDescriptions.xlsx', sheetname="Examples")
trainJson = train_df.to_json(orient='records')

# init the classifier and train it
myClassifier = TextClassifier(trainJson, 'description', 'segment')
   
# get all labels
print(myClassifier.getSegments())

# run prediction
to_predict_df = pd.read_excel('appDescriptions.xlsx', sheetname="Classify")
predictJson = to_predict_df.to_json(orient='records')
predicted_json = myClassifier.getBatchSegment(predictJson)

# save results to csv file
#predicted_df = pd.read_json(predicted_json)
#header = ["appId", "segment"]
#predicted_df.to_csv('output.csv', columns = header)

# get top segments for a specific row
predicted_labels = myClassifier.getTopSegments(to_predict_df.iloc[[1]].to_json(orient='records'), 3)