# Desafio Celero AI

### Project

This was made using Python 3.7, scikit-learn and NLTK

### Setup

Install the depencies

Using Pipenv

```sh
$ pipenv shell
$ pipenv install
```

Using pip

```sh
$ pip install -r requirements.txt
```

Download data packages for NLTK

```sh
$ ./setup.py
```

Change to src directory

```sh
$ cd src
```

To genarate the trainning data

```sh
src$ ./comment_classifier.py --train path_to_train_folder
```

To genarate the testing data

```sh
src$ ./comment_classifier.py --test path_to_test_folder
```

To run the classification on the test data (takes a long time)

```sh
src$ ./comment_classifier.py -c

or

src$ ./comment_classifier.py --classification
```

Run the classification over a single file and returns the predicted value

```sh
src$ ./comment_classifier.py -r path_to_file

or

src$ ./comment_classifier.py --run path_to_file
```