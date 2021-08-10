from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
import pandas
import math
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

# config
input_file_path = 'Data/mmkgap/machine_data_1.csv'
metrics = ['accuracy']
epochs = 1
batch_size = 5

dataframe = pandas.read_csv(input_file_path, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:-1].astype('float32')
Y = dataset[:, -1]

num_rows = X.shape[0]
num_cols = X.shape[1]

num_cols_padded_sqrt = math.ceil(math.sqrt(num_cols))

X.resize(int(num_rows * math.pow(num_cols_padded_sqrt, 2)))
X = X.reshape(-1, num_cols_padded_sqrt, num_cols_padded_sqrt, 1)


# train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2, random_state=13)


# baseline model
def create_baseline():
    # create model
    my_model = Sequential()
    my_model.add(
        Conv2D(64, kernel_size=(3, 3), activation='linear', input_shape=(num_cols_padded_sqrt, num_cols_padded_sqrt, 1),
               padding='same'))
    my_model.add(LeakyReLU(alpha=0.1))
    my_model.add(MaxPooling2D((2, 2), padding='same'))
    my_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    my_model.add(LeakyReLU(alpha=0.1))
    my_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    my_model.add(Conv2D(256, (3, 3), activation='linear', padding='same'))
    my_model.add(LeakyReLU(alpha=0.1))
    my_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    my_model.add(Flatten())
    my_model.add(Dense(512, activation='linear'))
    my_model.add(LeakyReLU(alpha=0.1))
    my_model.add(Dense(1, activation='sigmoid'))
    # Compile model
    my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return my_model


# estimator = KerasClassifier(build_fn=create_baseline, epochs=epochs, batch_size=batch_size, verbose=1,
#                             validation_data=(valid_X, valid_Y))

estimator = KerasClassifier(build_fn=create_baseline, epochs=epochs, batch_size=batch_size, verbose=1)

kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, X, Y, cv=kfold)

# results = cross_val_score(estimator, train_X, train_Y, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
