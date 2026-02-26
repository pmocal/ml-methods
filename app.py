from decimal import *
from pathlib import Path
import fileinput, random, math, os

from flask import Flask, render_template, request, redirect, url_for
from celery import Celery, Task, shared_task
import numpy, pickle
import scipy.io as sio

from sklearn.model_selection import train_test_split

# used with k-nearest neighbors implementation
DIGITS_TRAIN_LABELS = open('./digits-dataset/trainLabels.csv', 'r').readlines()
digits_train_features = []
with open('./digits-dataset/trainFeatures.csv', 'r') as train_features_old:
    for line in train_features_old:
        digits_train_features.append(numpy.array(tuple(map(float, line.split(',')))))

# used with neural network implementation
ITERATIONS = 1000000

# used with random forest implementation
MAX_DEPTH = 20

# used with decision tree and random forest implementation
data = sio.loadmat("./spam-dataset/spam_data.mat") #training_data and training_labels
spam_train_features = data["training_data"]
spam_train_labels = numpy.ravel(data["training_labels"])

def knn_majority(list):
    
    dict = {}
    for label, distance in list:
        if label not in dict:
            dict[label] = (1, [distance])
        else:
            dict[label] = (dict[label][0] + 1, dict[label][1] + distance)
    lad = sorted(dict.items(), key = lambda x: (-x[1][0], x[1][1]))
    return lad[0][0]

def majority(lst):
    if not lst:
        return 
    return max([0,1], key = lst.count)

def split(i, thresh, train_features, train_labels):
    left_features, left_labels, right_features, right_labels = [], [], [], []
    for x in range(len(train_features)): # not using whole feature set
        if train_features[x][i] <= thresh:
            left_features.append(train_features[x])
            left_labels.append(train_labels[x])
        else:
            right_features.append(train_features[x])
            right_labels.append(train_labels[x])
    return left_features, left_labels, right_features, right_labels

def entropy(count_0, count_1):
    count_0_frac = float(count_0)/(count_0 + count_1)
    count_1_frac = float(count_1)/(count_0 + count_1)
    log_0 = 0 if count_0_frac == 0 else count_0_frac * math.log(count_0_frac, 2)
    log_1 = 0 if count_1_frac == 0 else count_1_frac * math.log(count_1_frac, 2)
    return  -log_0 - log_1
    
def bagging(train_features, train_labels, k):
    k = int(math.ceil(k * len(train_labels))) # k can be anywhere between 0.7n and n
    bagged_features, bagged_labels = [None] * k, [None] * k	
    for i in range(k):
        j = int(random.random() * len(train_features))
        bagged_features[i] = train_features[j] #change parameter for build
        bagged_labels[i] = train_labels[j]
    return bagged_features, bagged_labels
    
def classify(features, tree):
    if tree[1] is None:
        return tree[0]
    elif features[tree[0]] <= tree[1]:
        return classify(features, tree[2])
    return classify(features, tree[3])
    
def build_decision_tree(train, depth, is_forest):
    (train_features, train_labels) = train
    if train_labels.count(train_labels[0]) == len(train_labels): # base case
        return (train_labels[0], None, None, None) # node format: (class in {0,1} if leaf else index of feature, None if leaf else threshhold, None if leaf else leftChild, None if leaf else rightChild)
    elif is_forest and depth == MAX_DEPTH:
        return (majority(train_labels), None, None, None)
    
    if is_forest:
        r = random.sample(range(len(train_features[0])), int(math.ceil(math.sqrt(len(train_features[0])))) + 5)
    else:
        r = range(32)
	
    info_gains = []
    for i in r:
        sorted_feat_i = sorted([(features[i], label) for features, label in zip(train_features, train_labels)])[::-1]
        # reverse sorted list of feature i for all training points grouped with label
        l_count_1 = sum([p[1] for p in sorted_feat_i])
        l_count_0 = len(train_labels) - l_count_1
        r_count_1 = 0
        r_count_0 = 0

        for index, (thresh, label) in enumerate(sorted_feat_i):
            if index > 0 and thresh != sorted_feat_i[index - 1][0]:
                l_coeff = float(l_count_0 + l_count_1)/len(sorted_feat_i)
                r_coeff = float(r_count_0 + r_count_1)/len(sorted_feat_i)
                info_gains.append(((l_coeff * entropy(l_count_0, l_count_1) + r_coeff * entropy(r_count_0, r_count_1)), i, thresh))
            r_count_1 += label
            r_count_0 += (1 - label)
            l_count_1 -= label
            l_count_0 -= (1 - label)
	
    if len(info_gains) == 0:
        return (majority(train_labels), None, None, None)
    info_gain, i, thresh = min(info_gains)
    left_features, left_labels, right_features, right_labels = split(i, thresh, train_features, train_labels)
    if not left_labels or not right_labels:
        return (majority(train_labels), None, None, None)
    left_child = build_decision_tree((left_features, left_labels), depth + 1, is_forest)
    right_child = build_decision_tree((right_features, right_labels), depth + 1, is_forest)
    return (i, thresh, left_child, right_child)

def build_random_forest(size, train_features, train_labels, is_forest):
    return [build_decision_tree(bagging(train_features, train_labels, 0.8), 0, is_forest) for i in range(size)]

class Neural_Network(object):
    
    def __init__(self, N_IN = 784, N_OUT = 10, N_HID = 200, EPSILON = 0.5):
        
        # haven't added bias
        self.input_layer_size = N_IN
        self.hidden_layer_size = N_HID
        self.output_layer_size = N_OUT

        # weights (parameters)
        self.W1 = numpy.random.randn(N_IN, N_HID) * EPSILON
        self.W2 = numpy.random.randn(N_HID, N_OUT) * EPSILON

    def forward(self, x):
        
        self.s2 = numpy.dot(x, self.W1)
        self.x2 = self.sigmoid(self.s2)
        self.s3 = numpy.dot(self.x2, self.W2)
        y_hat = self.sigmoid(self.s3)
        
        return y_hat

    def sigmoid(self, z):
        return 1 / (1 + numpy.exp(-z))

    def sigmoid_prime(self, z):
        return numpy.exp(-z) / ((1 + numpy.exp(-z)) ** 2)

    def tanh_prime(self, z):
        return 1 - numpy.tan(z) ** 2

    def cost_function_prime(self, X, y):
        
        i = random.randint(0, 47999)
        x_i = X[i]
        y_i = y[i]
        y_i_hat = self.forward(x_i)
		
        # mean squared error
        delta_3 = numpy.multiply(-(y_i - y_i_hat), self.sigmoid_prime(self.s3))
        dJdW2 = numpy.outer(self.x2, delta_3)
        
        delta_2 = numpy.dot(delta_3, self.W2.T) * self.sigmoid_prime(self.s2)
        dJdW1 = numpy.outer(x_i, delta_2)

        return dJdW1, dJdW2

    def update(self, train_features, train_labels, ETA):
        
        dJdW1, dJdW2 = self.cost_function_prime(train_features, train_labels)
        self.W1 = self.W1 - ETA * dJdW1
        self.W2 = self.W2 - ETA * dJdW2

def knn_validation():

    k = 10
    error_count = 0

    digits_val_labels = open('./digits-dataset/valLabels.csv', 'r').readlines()
    predicted_val_labels = []

    with fileinput.input(files=('./digits-dataset/valFeatures.csv')) as val_features:
        for val_feature_set in val_features:
            distances = []
            a = numpy.array(tuple(map(float, val_feature_set.split(','))))
            for j, train_feature_set in enumerate(digits_train_features):
                b = train_feature_set
                distances.append((j, numpy.linalg.norm(a - b)))
            distances = sorted(distances, key = lambda x: x[1])
            knn = []
            for l in range(0, k):
                label_index = distances[l][0]
                knn.append((int(DIGITS_TRAIN_LABELS[label_index]), distances[l][1]))
            prediction = knn_majority(knn)
            if prediction != int(digits_val_labels[fileinput.lineno()-1]):
                error_count += 1
            predicted_val_labels.append(prediction)
        return error_count, len(predicted_val_labels), len(digits_train_features), k
        
def dt_validation():
    num_trees = 1
    train_features, val_features, train_labels, val_labels = train_test_split(spam_train_features, spam_train_labels, test_size = 0.4)
    error = 0
    forest = build_random_forest(num_trees, train_features, train_labels, False)
    pred_val_labels = []
    for features in val_features:
        votes = []
        for tree in forest:
            votes.append(classify(features, tree))
        prediction = majority(votes)
        pred_val_labels.append(prediction)
    return sum([1 for i in range(len(val_labels)) if val_labels[i] != pred_val_labels[i]]), len(val_labels), len(train_features), num_trees, MAX_DEPTH

def rf_validation():
    num_trees = 50
    train_features, val_features, train_labels, val_labels = train_test_split(spam_train_features, spam_train_labels, test_size = 0.4)
    error = 0
    forest = build_random_forest(num_trees, train_features, train_labels, True)
    pred_val_labels = []
    for features in val_features:
        votes = []
        for tree in forest:
            votes.append(classify(features, tree))
        prediction = majority(votes)
        pred_val_labels.append(prediction)
    return sum([1 for i in range(len(val_labels)) if val_labels[i] != pred_val_labels[i]]), len(val_labels), len(train_features), num_trees, MAX_DEPTH

@shared_task(time_limit=4000)
def nn_validation():
    train = sio.loadmat("./digits-dataset/train.mat") #training_data and training_labels
    train_features = train["train_images"]
    train_features = numpy.reshape(train_features, (784, 60000)).T
    train_labels = numpy.ravel(train["train_labels"])
    train_features = train_features / 256.0
    new_train_labels = []
    for label in train_labels:
        new_label = numpy.zeros(10)
        new_label[label] = 1
        new_train_labels.append(new_label)
    train_labels = numpy.array(new_train_labels)

    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size = 0.2)
    train = Neural_Network()
    validate = Neural_Network()
    
    for j in range(ITERATIONS):
        if j % 1000 == 0:
            validate.W1 = train.W1
            validate.W2 = train.W2
            pred_val_labels = validate.forward(val_features)
            labels = numpy.argmax(pred_val_labels, axis = 1)
            pred_val_labels = []
            for label in labels:
                new_label = numpy.zeros(10)
                new_label[label] = 1
                pred_val_labels.append(new_label)
            pred_val_labels = numpy.array(pred_val_labels)
            correct = sum([1 for i in range(len(val_labels)) if (val_labels[i] == pred_val_labels[i]).all()])/float(len(val_labels))	
            pred_train_labels = train.forward(train_features)
            labels = numpy.argmax(pred_train_labels, axis = 1)
            pred_train_labels = []
            for label in labels:
                new_label = numpy.zeros(10)
                new_label[label] = 1
                pred_train_labels.append(new_label)
            pred_train_labels = numpy.array(pred_train_labels)
            correct = sum([1 for i in range(len(train_labels)) if (train_labels[i] == pred_train_labels[i]).all()])/float(len(train_labels))	
        train.update(train_features, train_labels, 0.05)
        
    return sum([1 for i in range(len(val_labels)) if (val_labels[i] == pred_val_labels[i]).all()]), len(val_labels), len(train_features), ITERATIONS

def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)
    celery_app = Celery(app.name)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.Task = FlaskTask
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app

app = Flask(__name__)
app.config.from_mapping(
    CELERY=dict(
        broker_url=os.environ['CLOUDAMQP_URL'],
        result_backend='rpc://',
        task_ignore_result=False,
    ),
)
celery_app = celery_init_app(app)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if(request.form.get('ml') == 'spam'):
            return redirect(url_for('switch'))
        return redirect(url_for('default_get'))
    return render_template("index.html")

@app.get("/digits")
def default_get():
    numpy.set_printoptions(precision = 2, floatmode = "fixed")
    n = numpy.random.randint(len(digits_train_features))
    return render_template("digits.html", example = ([numpy.array([digits_train_features[n][i] for i in range(r, len(digits_train_features[0]), 28)]) for r in range(28)], DIGITS_TRAIN_LABELS[n]))

@app.post("/digits")
def default_post():
    if request.form.get('alg') == 'nn':
        app.g = nn_validation.delay()
        return render_template("nn.html", str = f"Your task (ID {app.g.id}) will take up to half an hour. Clicking the button below will retrieve the result if it's ready.")
    elif request.form.get('alg') == 'knn':
        return render_template("results.html", validation = knn_validation, algorithm = { 'knn': True })
    else:
        return render_template("nn.html", ready = app.g.ready(), str = f"We predicted correctly {(app.g.result[0] / app.g.result[1]) * 100}% of the time on a validation set of {app.g.result[1]} examples. \
            Our dataset is the MNIST database and we trained on {app.g.result[2]} examples with {app.g.result[3]} iterations." if app.g.ready() else f"Task ID {app.g.id} not yet ready, please wait longer. \
            Clicking the button below will retrieve the result if it's ready.")
 
@app.route("/email", methods=['GET','POST'])
def switch():
    if request.method == 'POST':
        if(request.form.get('alg') == 'rf'):
            return render_template("results.html", validation = rf_validation, algorithm = { 'rf': True })
        return render_template("results.html", validation = dt_validation, algorithm = { 'dt': True })
    spam = random.randint(0, 1)
    if spam:
        return render_template("email.html", example = (random.choice(list(Path('./spam-dataset/spam/').iterdir())).read_text(), 'Spam'))
    else:
        return render_template("email.html", example = (random.choice(list(Path('./spam-dataset/ham/').iterdir())).read_text(), 'Genuine email'))