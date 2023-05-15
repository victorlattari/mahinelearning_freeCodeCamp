import itertools
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Creating a dictionary to keep track of play order counts
combinations = list(itertools.product(["R", "P", "S"], repeat=2))
play_order = {"".join(comb): 0 for comb in combinations}

# Mapping the ideal and the counter response for each play
ideal_response = {"P": "S", "R": "P", "S": "R"}
counter_response = {"P": "R", "R": "S", "S": "P"}

# Initializing empty lists to store previous plays
my_last, my_history, op_history = [], [], []
guess_move, bot = 0, 0


def player(prev_play, opponent_history=[]):
    global bot

    # Resetting the variables when playing against a new bot
    if len(op_history) % 1000 == 0:
        reset()
        bot += 1

    opp_prev = convert_index(prev_play)
    op_history.append(opp_prev)

    accuracy = 0
    guess = "R"
    pred = "R"

    # Playing against quincy
    # Using KNN model to predict opponent's next move
    if bot == 1 and len(op_history) > 30:
        accuracy, guess = knn_model(op_history)

    # Playing against abbey
    # Using a double sequence prediction for guess the next move
    elif bot == 2:
        if len(my_last) > 1:
            guess, pred = seq(my_last[-1], my_last)
            my_last.append(pred)
        else:
            my_last.extend(["R", pred])

    # Playing against kris
    # Using the counter response in the function of my last move
    elif bot == 3 and len(op_history) > 30:
        guess = counter_response[my_history[-1]]

    # Playing against mrugesh
    # Using KNN model to predict opponent's next move
    elif bot > 3 and len(op_history) > 30:
        accuracy, guess = knn_model(op_history)

    my_history.append(guess)

    return guess


def knn_model(op_history):
    qt_pred = 5

    # Creating a matrix to predict the next move from the last 5 moves
    guess_matrix = np.empty((len(op_history) - qt_pred, qt_pred))

    for i in range(len(op_history) - qt_pred):
        guess_matrix[i] = op_history[i : i + qt_pred]

    # Transforming the matrix into a pandas DataFrame
    df = pd.DataFrame(guess_matrix)[-100:]

    # Creating the train, valid, and test datasets
    train, valid, test = np.split(
        df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))]
    )

    X_train = train[train.columns[:-1]].values
    y_train = train[train.columns[-1]].values

    X_valid = valid[valid.columns[:-1]].values
    y_valid = valid[valid.columns[-1]].values

    X_test = test[test.columns[:-1]].values
    y_test = test[test.columns[-1]].values

    # Creating a KNN model with 5 neighbors
    knn_model = KNeighborsClassifier(n_neighbors=5)

    # Training the KNN model
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

    # Calculating the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)

    # Predicting the next move based on the last 4 moves
    last_moves = np.reshape(np.array(op_history[-4:]), (1, -1))
    y_pred = knn_model.predict(last_moves)

    pred_knn = "P" if y_pred == 1 else "S" if y_pred == 2 else "R"

    # Returning the accuracy and the predicted move
    return accuracy, pred_knn


def seq(prev_play, my_last):
    # Extracting the last 2 moves and joining them into a string
    last_moves = "".join(my_last[-2:])

    # Incrementing the count for the specific move combination
    if len(last_moves) == 2:
        play_order[last_moves] += 1

        prediction = max(
            ["".join([prev_play, move]) for move in ["R", "P", "S"]], key=play_order.get
        )[-1:]

        # Returning the ideal and counter responses based on the predicted opponent's move
        return ideal_response[prediction], counter_response[prediction]


def convert_index(play):
    # Mapping of plays to their corresponding indexes
    index_map = {"R": 1, "P": 2, "S": 3}
    return index_map.get(play, 0)


def reset():
    # Resetting the play_order dictionary
    for key in play_order:
        play_order[key] = 0

    # Clearing the lists
    my_last.clear()
    op_history.clear()
    my_history.clear()