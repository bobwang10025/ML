import numpy as np

def documents(speechdoc):
	return list(speechdoc.reviews())
def continuous(speechdoc):
	return list(speechdoc.scores())
def make_categorical(speechdoc):
	"""
	terrible:   0.0 <y <= 3.0
	ok:          3.0 <y <= 5.0
	great:      5.0 <y <= 7.0
	amazing: 7.0 <y <= 10.1
	"""
	return np.digitize(continuous(speechdoc), [0.0, 3.0, 5.0, 7.0, 10.1])

from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

def train_model(path, model, continuous=True, saveto=None, cv=12):
	"""
	train model, construct cross-validation scores, fit model and
	return scores
	"""
	# load data and label it
	speechdoc = PickledReviewsReader(path)
	X = documents(speechdoc)
	if continuous:
		y = continuous(speechdoc)
		scoring = 'r2_score'
	else:
		y = make_categotical(speechdoc)
		scoring = 'f1_score'
	# compute scores
	scores = cropss_val_score(model, X, y, cv=cv, scoring=scoring)
	# save it
	if storage_disk:
		joblib.dump(model, storage_disk)
	# fit model
	model.fit(X,y)
	# return scores
	return scores

if __name__ == '__main__':
	from transformer import TextNormalizer
	from reader import PickledReviewsReader
	
	from sklearn.pipeline import Pipeline		
	from sklearn.neural_network import MLPRegressor, MLPClassifier
	from sklearn.feature_extraction.text import TfidfVectorizer
	
	# Path to post and pre processed
	spath = '../review_speechdoc_proc'

	regressor = Pipeline([
		('norm', TextNormalizer()),
		('tfidf', TfidfVectorizer()),
		('ann', MLPRegressor(hiddel_layer_sizes=[500,150], verbose=True))
	])
	regression_scores = train_model(spath, regressor, continuous=True)
	
	classifier = Pipeline([
		('norm', TextNormalizer()),
		('tfidf', TfidfVectorizer()),
		('ann', MLPClassifier(hiddel_layer_sizes=[500,150], verbose=True))
	])
	classifier_scores = train_model(spath, classifier, continuour=False)

