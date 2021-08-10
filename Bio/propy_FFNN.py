import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# config
input_file_path = 'Data/propy/propy_machine_data_1.csv'
metrics = ['accuracy']
epochs = 1
batch_size = 5

# load dataset
dataframe = pandas.read_csv(input_file_path, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:-1].astype(float)
Y = dataset[:, -1]

num_cols = X.shape[1]


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(512, input_dim=num_cols, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return model


estimator = KerasClassifier(build_fn=create_baseline, epochs=epochs, batch_size=batch_size, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
