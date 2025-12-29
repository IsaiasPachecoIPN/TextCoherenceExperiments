import os
import coloredlogs, logging
import pickle
import math
import re
import torch
import optuna
import pandas                           as pd
import numpy                            as np
import matplotlib.pyplot                as plt
import matplotlib.gridspec              as gridspec
import matplotlib.colors                as mcolors
import matplotlib.cm                    as cm
import gensim.downloader                as api
import matplotlib.image                 as mpimg
import seaborn                          as sns
import tempfile

from datetime                           import datetime
from sklearn.base                       import clone
from sklearn.model_selection            import train_test_split
from sklearn.model_selection            import cross_val_score
from sklearn.model_selection            import KFold
from sklearn.model_selection            import StratifiedKFold

from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.feature_extraction.text    import TfidfVectorizer

from sklearn.multioutput                import MultiOutputRegressor
from sklearn.multioutput                import MultiOutputClassifier

from sklearn.linear_model               import LogisticRegression
from sklearn.linear_model               import RidgeClassifier
from sklearn.linear_model               import RidgeClassifierCV
from sklearn.linear_model               import SGDClassifier

from sklearn.ensemble                   import RandomForestRegressor
from sklearn.ensemble                   import RandomForestClassifier
from sklearn.ensemble                   import GradientBoostingClassifier
from sklearn.ensemble                   import AdaBoostClassifier

from sklearn.naive_bayes                import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors                  import KNeighborsClassifier
from sklearn.tree                       import DecisionTreeClassifier

from sklearn.svm                        import SVC
from sklearn.svm                        import LinearSVC
from xgboost                            import XGBClassifier
from lightgbm                           import LGBMClassifier

from sklearn.metrics                    import mean_squared_error
from sklearn.metrics                    import accuracy_score
from sklearn.metrics                    import recall_score
from sklearn.metrics                    import precision_score
from sklearn.metrics                    import classification_report
from sklearn.metrics                    import confusion_matrix
from sklearn.metrics                    import f1_score
from sklearn.metrics                    import log_loss
from sklearn.metrics                    import confusion_matrix
from sklearn.metrics                    import hamming_loss
from sklearn.metrics                    import f1_score
from sklearn.metrics                    import precision_score

from sklearn.neural_network             import MLPRegressor
from sklearn.neural_network             import MLPClassifier
from imblearn.under_sampling            import RandomUnderSampler

from gensim.models                     import FastText
from gensim.models                     import word2vec
from gensim.utils                      import simple_preprocess

from transformers                       import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoConfig
from transformers                       import EarlyStoppingCallback
from torch.utils.data                   import Dataset, DataLoader

from optuna.samplers                    import TPESampler
from optuna.pruners                     import MedianPruner
from optuna.pruners                     import HyperbandPruner

from transformers                       import TrainerCallback

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt="%(levelname)s[Line:%(lineno)d]:%(message)s")

class NLP_Model:

    def __init__(self, output_dir='./output'):
        self.data                   = None
        self.vocabulary             = None
        self.word_count_dataset     = None
        self.tfidf_dataset          = None
        self.word_count_model       = None
        self.single_output_model    = None
        self.vectorizer_model       = None
        self.embedding_dim          = 300
        self.output_dir             = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_csv(self, path, verbose=False):

        """
        Load the dataset from a csv file
        @param path: path to the csv file
        @param verbose: print additional information
        """

        self.data = pd.read_csv(path)

        print(f"Dataset loaded")

        if verbose:
            print(f'Dataset shape: {self.data.shape}')

    def build_vocabulary(self, verbose=False, override=False):

        """
        Build the vocabulary from the dataset. If the vocabulary already exists, it will be loaded.
        @param verbose: print additional information
        @param override: override the existing vocabulary
        """

        #Check if vocabulary already exists
        try:
            if override:
                    raise Exception("Override")
            with open(os.path.join(self.output_dir, 'vocabulary.pkl'), 'rb') as f:
                self.vocabulary = pickle.load(f)

            print(f'Vocabulary loaded')

        except:
            self.vocabulary = set()
            for text in self.data['text']:
                words = text.split()
                words = [word.lower().strip().replace("\n", "") for word in words]
                self.vocabulary.update(words)

            #save vocabulary
            with open(os.path.join(self.output_dir, 'vocabulary.pkl'), 'wb') as f:
                pickle.dump(self.vocabulary, f)

            print(f'Vocabulary created')

        if verbose:
            print(f'Vocabulary: {self.vocabulary}')
            print(f'Vocabulary size: {len(self.vocabulary)}')


    def preprocess_text(self, target_label, lower=False):
        """
        Preprocess the target_label using the flag
        """
        if lower:
            self.data[target_label] = self.data[target_label].apply(lambda x: x.lower() if isinstance(x, str) else x)

        # Remove specific special characters, including JSON problematic characters (like ", \, [], <>, {}, etc.)
        self.data[target_label] = self.data[target_label].apply(
            lambda x: re.sub(r'[\[\]<>\"\\\{\}]', '', x) if isinstance(x, str) else x
        )

        logger.info("Preprocessing finished")

    def join_text_columns(self, columns, new_column_name, separator='-'):
        """
        Join multiple text columns into a single one

        Parameters:
        - columns: list of column names to join
        - new_column_name: name for the new joined column
        - separator: string to use between joined values
        """

        missing_columns = [col for col in columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"The following columns were not found in the DataFrame: {missing_columns}")

        # Create a copy of the data to avoid modifying the original DataFrame
        bk_dataframe = pd.DataFrame()

        # Create new dataframe with the joined text
        bk_dataframe[new_column_name] = self.data[columns].apply(
            lambda x: separator.join(x.astype(str)),  # Convert to string in case there are non-string values
            axis=1
        )

        # Add the rest of the columns
        bk_dataframe = pd.concat([bk_dataframe, self.data.drop(columns=columns)], axis=1)

        # Reset index
        bk_dataframe.reset_index(drop=True, inplace=True)

        self.data = bk_dataframe

        print(bk_dataframe.head())




    def preprocess_text_dataframe(self, dataframe, target_label, lower=False):
        """
        Preprocess the target_label using the flag
        """
        if lower:
            dataframe[target_label] = dataframe[target_label].apply(lambda x: x.lower() if isinstance(x, str) else x)

        # Remove specific special characters, including JSON problematic characters (like ", \, [], <>, {}, etc.)
        dataframe[target_label] = dataframe[target_label].apply(
            lambda x: re.sub(r'[\[\]<>\"\\\{\}]', '', x) if isinstance(x, str) else x
        )

        logger.info("Preprocessing finished")

    def build_word_count_vectorizer(self, verbose=False, override=False, ngram_range=(1, 1)):

        """
        Build the word count vectorizer using the vocabulary
        """

        #Check if the dataset already exists
        try:
            if override:
                raise Exception("Override")
            with open(os.path.join(self.output_dir, 'word_count_dataset.pkl'), 'rb') as f:
                self.word_count_dataset = pickle.load(f)
            with open(os.path.join(self.output_dir, 'vectorizer_model.pkl'), 'rb') as f:
                self.vectorizer_model = pickle.load(f)
            print(f'Dataset loaded')
        except:
            corpus = self.data['text']
            vectorizer = CountVectorizer(vocabulary=list(self.vocabulary), ngram_range=ngram_range)

            X = vectorizer.fit_transform(corpus)

            print(f"Corpus length: {len(corpus)}")
            print(f"Vocabulary length: {len(self.vocabulary)}")
            print(f"X shape: {X.shape}")

            self.word_count_dataset = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

            #Adding the rest of the columns
            self.word_count_dataset['[SCORE]'] = self.data['score']

            # Reset index
            self.word_count_dataset.reset_index(drop=True, inplace=True)
            self.data.reset_index(drop=True, inplace=True)  # Ensure alignment

            #Save the dataset using pickle
            self.word_count_dataset.to_pickle(os.path.join(self.output_dir, 'word_count_dataset.pkl'))
            logger.info(f'Dataset created')

            #Save the vectorizer using pickle
            self.vectorizer_model = vectorizer
            with open(os.path.join(self.output_dir, 'vectorizer_model.pkl'), 'wb') as f:
                pickle.dump(vectorizer, f)

        print(f'Dataframe {self.word_count_dataset.head()}')
        print(f'Dataframe shape: {self.word_count_dataset.shape}')


    def build_tfidf_vectorizer(self, verbose=False, override=False, ngram_range=(1, 3)):

        """
        Build the word count vectorizer using the vocabulary
        """

        #Check if the dataset already exists
        try:
            if override:
                raise Exception("Override")
            with open(os.path.join(self.output_dir, 'tfidf_dataset.pkl'), 'rb') as f:
                self.word_count_dataset = pickle.load(f)
            with open(os.path.join(self.output_dir, 'vectorizer_model.pkl'), 'rb') as f:
                self.vectorizer_model = pickle.load(f)

            logger.info(f'Dataset loaded')
            logger.info(f'TFIDF model loaded')
        except:
            corpus = self.data['text']

            vectorizer = TfidfVectorizer(vocabulary=list(self.vocabulary), ngram_range=ngram_range)

            X = vectorizer.fit_transform(corpus)

            print(f"Corpus length: {len(corpus)}")
            print(f"Vocabulary length: {len(self.vocabulary)}")
            print(f"X shape: {X.shape}")

            self.word_count_dataset = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

            #Adding the rest of the columns
            self.word_count_dataset['[SCORE]'] = self.data['score']

            # Reset index
            self.word_count_dataset.reset_index(drop=True, inplace=True)
            self.data.reset_index(drop=True, inplace=True)  # Ensure alignment

            #Save the dataset using pickle
            self.word_count_dataset.to_pickle(os.path.join(self.output_dir, 'tfidf_dataset.pkl'))
            #Save the vectorizer using pickle
            self.vectorizer_model = vectorizer
            with open(os.path.join(self.output_dir, 'vectorizer_model.pkl'), 'wb') as f:
                pickle.dump(vectorizer, f)

            logger.info(f'Dataset created')
            logger.info(f'TFIDF model created')


        print(f'Dataframe {self.word_count_dataset.head()}')
        print(f'Dataframe shape: {self.word_count_dataset.shape}')

    def build_fasttext_vectorizer(self, verbose=False, override=False):
        """
        Build the FastText vectorizer using a pretrained model.
        """

        try:
            if override:
                raise Exception("Override")
            with open(os.path.join(self.output_dir, 'fasttext_dataset.pkl'), 'rb') as f:
                self.fasttext_dataset = pickle.load(f)
            with open(os.path.join(self.output_dir, 'fasttext_model.pkl'), 'rb') as f:
                self.fasttext_model = pickle.load(f)

            logger.info(f'FastText dataset loaded')
            logger.info(f'FastText model loaded')

        except:
            corpus = self.data['text']

            # Load pretrained FastText model (from gensim)
            logger.info("Loading pretrained FastText model...")
            fasttext = api.load("fasttext-wiki-news-subwords-300")

            def text_to_vector(text):
                words = text.split()  # Tokenization
                vectors = [fasttext[word] for word in words if word in fasttext]
                return np.mean(vectors, axis=0) if vectors else np.zeros(300)  # Handle empty cases

            # Convert each text entry into a FastText vector
            X = np.vstack(corpus.apply(text_to_vector))

            print(f"Corpus length: {len(corpus)}")
            print(f"X shape: {X.shape}")

            # Create DataFrame with embeddings
            self.fasttext_dataset = pd.DataFrame(X)

            # Adding the rest of the columns
            self.fasttext_dataset['[SCORE]'] = self.data['score']

            # Reset index
            self.fasttext_dataset.reset_index(drop=True, inplace=True)
            self.data.reset_index(drop=True, inplace=True)  # Ensure alignment

            # Set the dataset to be used in the next steps
            self.word_count_dataset =  self.fasttext_dataset

            # Save dataset using pickle
            self.fasttext_dataset.to_pickle(os.path.join(self.output_dir, 'fasttext_dataset.pkl'))

            # Save vectorizer using pickle
            self.vectorizer_model = fasttext
            with open(os.path.join(self.output_dir, 'fasttext_model.pkl'), 'wb') as f:
                pickle.dump(fasttext, f)

            logger.info(f'FastText dataset created')
            logger.info(f'FastText model loaded')

        if verbose:
            print(f'Dataframe {self.fasttext_dataset.head()}')
            print(f'Dataframe shape: {self.fasttext_dataset.shape}')



    def build_word2vect_vectorizer(self, verbose=False, override=False):
        """
        Build the Word2Vec vectorizer using a pretrained model.
        """

        try:
            if override:
                raise Exception("Override")
            with open(os.path.join(self.output_dir, 'word2vec_dataset.pkl'), 'rb') as f:
                self.word2vec_dataset = pickle.load(f)
                self.word_count_dataset = self.word2vec_dataset
            with open(os.path.join(self.output_dir, 'word2vec_model.pkl'), 'rb') as f:
                self.word2vec_model = pickle.load(f)
                self.vectorizer_model = self.word2vec_model
                
                print(f"Model type: {type(self.word2vec_model)}")

            logger.info(f'Word2Vec dataset loaded')
            logger.info(f'Word2Vec model loaded')

        except:
            corpus = self.data['text']

            # Load pretrained Word2Vec model (Google News 300D)
            logger.info("Loading pretrained Word2Vec model...")
            word2vec = api.load("word2vec-google-news-300")

            def text_to_vector(text):
                words = text.split()  # Tokenization
                vectors = [word2vec[word] for word in words if word in word2vec]
                return np.mean(vectors, axis=0) if vectors else np.zeros(300)  # Handle empty cases

            # Convert each text entry into a Word2Vec vector
            X = np.vstack(corpus.apply(text_to_vector))

            print(f"Corpus length: {len(corpus)}")
            print(f"X shape: {X.shape}")

            # Create DataFrame with embeddings
            self.word2vec_dataset = pd.DataFrame(X)

            # Adding the rest of the columns
            self.word2vec_dataset['[SCORE]'] = self.data['score']

            # Reset index
            self.word2vec_dataset.reset_index(drop=True, inplace=True)
            self.data.reset_index(drop=True, inplace=True)  # Ensure alignment

            self.word_count_dataset = self.word2vec_dataset

            # Save dataset using pickle
            self.word2vec_dataset.to_pickle(os.path.join(self.output_dir, 'word2vec_dataset.pkl'))

            # Save vectorizer using pickle
            self.vectorizer_model = word2vec
            with open(os.path.join(self.output_dir, 'word2vec_model.pkl'), 'wb') as f:
                pickle.dump(word2vec, f)

            logger.info(f'Word2Vec dataset created')
            logger.info(f'Word2Vec model loaded')
            
            print(f"Model type: {type(word2vec)}")

        if verbose:
            print(f'Dataframe {self.word2vec_dataset.head()}')
            print(f'Dataframe shape: {self.word2vec_dataset.shape}')

    def build_glove_vectorizer(self, verbose=False, dim='100', override=False):
        """
        Build the GloVe vectorizer using a pretrained GloVe model from Gensim's API.
        """

        try:
            if override:
                raise Exception("Override")
            with open(os.path.join(self.output_dir, 'glove_dataset.pkl'), 'rb') as f:
                self.glove_dataset = pickle.load(f)
            with open(os.path.join(self.output_dir, 'glove_model.pkl'), 'rb') as f:
                self.glove_model = pickle.load(f)

            logger.info(f'GloVe dataset loaded')
            logger.info(f'GloVe model loaded')

        except:
            corpus = self.data['text']

            # Load pretrained GloVe model (from Gensim's API)
            logger.info("Loading pretrained GloVe model...")
            glove = api.load(f"glove-wiki-gigaword-{dim}")  # You can also choose other GloVe models with different dimensions

            def text_to_vector(text):
                words = text.split()  # Simple whitespace tokenization, can be improved with NLTK or spaCy
                vectors = [glove[word] for word in words if word in glove]
                return np.mean(vectors, axis=0) if vectors else np.zeros(int(dim))  # Return zero vector if no words are in GloVe

            # Convert each text entry into a GloVe vector
            X = np.vstack(corpus.apply(text_to_vector))

            print(f"Corpus length: {len(corpus)}")
            print(f"X shape: {X.shape}")

            # Create DataFrame with embeddings
            self.glove_dataset = pd.DataFrame(X)

            # Adding the rest of the columns
            self.glove_dataset['[SCORE]'] = self.data['score']

            # Reset index
            self.glove_dataset.reset_index(drop=True, inplace=True)
            self.data.reset_index(drop=True, inplace=True)  # Ensure alignment

            self.word_count_dataset = self.glove_dataset

            # Save dataset using pickle
            self.glove_dataset.to_pickle(os.path.join(self.output_dir, 'glove_dataset.pkl'))

            # Save vectorizer using pickle
            self.vectorizer_model = glove
            with open(os.path.join(self.output_dir, 'glove_model.pkl'), 'wb') as f:
                pickle.dump(glove, f)

            logger.info(f'GloVe dataset created')
            logger.info(f'GloVe model loaded')

        if verbose:
            print(f'Dataframe {self.glove_dataset.head()}')
            print(f'Dataframe shape: {self.glove_dataset.shape}')

    def balance_dataset_for_singleOutput(self, target_label):

        print(f'Number of labels before balancing: {self.data[target_label].sum()}')

        rus = RandomUnderSampler(random_state=42)

        target_data = self.data[target_label]
        X_data = self.data.drop(columns=[target_label])

        X_resampled, y_resampled = rus.fit_resample(X_data, target_data)

        dataframe = pd.DataFrame(X_resampled, columns=X_data.columns)
        dataframe[target_label] = y_resampled

        dataframe.reset_index(drop=True, inplace=True)

        print(f'Number of labels after balancing: {dataframe[target_label].sum()}')

        self.data = dataframe


    def get_default_models(self, random_state=33):
        """
        Returns a list of default models to be used in the classifier.
        """
        return [
            LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=random_state),
            LogisticRegression(solver='liblinear', C=0.5, penalty='l1', random_state=random_state),
            SGDClassifier(loss='log_loss', penalty='l1', max_iter=1500, n_jobs=8, early_stopping=True, random_state=random_state),
            SGDClassifier(loss='hinge', penalty='l1', max_iter=1500, n_jobs=8, early_stopping=True, random_state=random_state),
            SGDClassifier(loss='hinge', penalty='l2', max_iter=1500, n_jobs=8, early_stopping=True, random_state=random_state),
            SGDClassifier(loss='modified_huber', penalty='l1', max_iter=1500, n_jobs=8, early_stopping=True, random_state=random_state),
            SGDClassifier(loss='modified_huber', penalty='l2', max_iter=1500, n_jobs=8, early_stopping=True, random_state=random_state),
        ]

    def build_singleOutputClassifier(
        self,
        models=None,
        verbose=False,
        override=False,
        embedding_name='word_count',
        metric='accuracy',
        random_state=33
    ):
        """
        Build the model to predict the score using the word count dataset.
        :param metric: Métrica principal para seleccionar el mejor modelo.
                    Opciones típicas: 'accuracy', 'f1', 'precision'.
        """
        
        try:
            if override:
                raise Exception("Override")
            with open(os.path.join(self.output_dir, 'single_output_model.pkl'), 'rb') as f:
                model = pickle.load(f)
                self.single_output_model = model
            logger.info(f'Model loaded: Number of characteristics: {len(self.single_output_model.coef_[0])}')

        except:

            print(f"Dataset columns {self.word_count_dataset.columns}")

            data = self.word_count_dataset.drop(columns=['[SCORE]'])
            targets = self.word_count_dataset['[SCORE]']

            logger.info(f"Data to train shape: {data.shape}")

            # Split para un futuro test (si quieres ver métricas fuera del loop de validación cruzada)
            X_train, X_test, y_train, y_test = train_test_split(
                data, targets, test_size=0.2, random_state=42
            )

            print(f"Training data shape: {X_train.shape}")
            print(f"Test data shape: {X_test.shape}")

            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            if models is None:
                models = self.get_default_models(random_state=random_state)

            skip_idx = []

            results = []
            best_score = -1  # Valor inicial bajo para poder comparar
            best_model = None

            # Entrenamiento de todos los modelos con K-Fold
            for idx, model in enumerate(models):
                if idx in skip_idx:
                    continue

                model_name = type(model).__name__
                if verbose:
                    print(f"Training model {model_name}")

                # Guardamos métricas de cada fold
                fold_scores = {
                    'Accuracy': [],
                    'F1': [],
                    'Precision': []
                }

                for train_index, val_index in kf.split(data, targets):
                    X_train_fold, X_val_fold = data.iloc[train_index], data.iloc[val_index]
                    y_train_fold, y_val_fold = targets.iloc[train_index], targets.iloc[val_index]

                    clf = model.fit(X_train_fold, y_train_fold)
                    y_pred_fold = clf.predict(X_val_fold)

                    fold_scores['Accuracy'].append(accuracy_score(y_val_fold, y_pred_fold))
                    fold_scores['F1'].append(f1_score(y_val_fold, y_pred_fold, average="weighted"))
                    fold_scores['Precision'].append(precision_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0))

                # Calculamos la media de cada métrica en los folds
                mean_accuracy = np.mean(fold_scores['Accuracy'])
                mean_f1 = np.mean(fold_scores['F1'])
                mean_precision = np.mean(fold_scores['Precision'])

                # Registramos resultados en un dict
                model_results = {
                    'Model': model_name,
                    'Mean Accuracy': round(mean_accuracy, 3),
                    'Mean F1': round(mean_f1, 3),
                    'Mean Precision': round(mean_precision, 3)
                }

                print(f"Model results: {model_results}")

                results.append(model_results)

                # Escogemos la métrica que usaremos para elegir el "mejor modelo"
                if metric == 'accuracy':
                    chosen_metric_value = mean_accuracy
                elif metric == 'f1':
                    chosen_metric_value = mean_f1
                elif metric == 'precision':
                    chosen_metric_value = mean_precision
                else:
                    # Por defecto, si envían algo no contemplado, usaremos accuracy
                    chosen_metric_value = mean_accuracy

                # Comparamos contra el mejor puntaje global hasta el momento
                if chosen_metric_value > best_score:
                    best_score = chosen_metric_value
                    best_model = clone(model)  # O puedes guardar directamente 'clf'

            # Imprimimos tabla de resultados
            df_results = pd.DataFrame(results)
            print(df_results.to_string())

            # Entrenamos el best_model con TODOS los datos de entrenamiento
            # para tener un modelo final listo para test o para producción.
            self.single_output_model = best_model.fit(X_train, y_train)

            print(f"\nBest model: {type(best_model).__name__}")
            print(f"Best {metric}: {best_score:.3f}")

            # Guardamos los resultados en CSV
            df_results.to_csv('model_comparison.csv', index=False)

            # --- Evaluamos en el test set (fuera de folds) ---
            y_pred = self.single_output_model.predict(X_test)

            # Generamos reporte
            self.generate_classification_report(y_test, y_pred, {}, f'{type(self.single_output_model).__name__}_{embedding_name}')

            # Mostramos métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted')

            print(f"\nMetrics on final test set:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"F1 Score: {f1:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")

            print(classification_report(y_test, y_pred, zero_division=0))

            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=np.unique(targets), yticklabels=np.unique(targets))
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()

            # Guardamos el modelo entrenado
            with open(os.path.join(self.output_dir, 'single_output_model.pkl'), 'wb') as f:
                pickle.dump(self.single_output_model, f)

            print(f'Model created and saved.')


    def get_classifier_test_score(self, test_dataset_path, embedding_name='word_count', verbose=False):

        # Cargar el dataset
        test_dataframe = pd.read_csv(test_dataset_path)

        # Preprocesamiento
        self.preprocess_text_dataframe(test_dataframe, 'text', lower=True)

        # Print vectorizer type
        if verbose:
            print(f"Vectorizer type: {self.vectorizer_model}")
            print(f"Model type: {type(self.single_output_model)}")

        # Vectorización del texto
        if embedding_name in ['word_count', 'tfidf']:
            vectorizer = self.vectorizer_model
            X_test = vectorizer.transform(test_dataframe['text'])
        elif embedding_name in ['fasttext', 'word2vec']:
            def text_to_vector(text):
                words = text.split()
                vectors = [self.vectorizer_model[word] for word in words if word in self.vectorizer_model]
                return np.mean(vectors, axis=0) if vectors else np.zeros(300)  # Manejo de casos vacíos

            X_test = np.vstack(test_dataframe['text'].apply(text_to_vector))
        elif embedding_name == 'glove':
            def text_to_vector(text):
                words = text.split()
                vectors = [self.vectorizer_model[word] for word in words if word in self.vectorizer_model]
                return np.mean(vectors, axis=0) if vectors else np.zeros(self.embedding_dim)  # Manejo de casos vacíos

            X_test = np.vstack(test_dataframe['text'].apply(text_to_vector))
        else:
            raise Exception("Invalid embedding name")

        # Obtener etiquetas reales y predicciones
        y_test = test_dataframe['score'].to_numpy()
        y_pred = self.single_output_model.predict(X_test)

        # Calcular matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Identificar TP, FP, FN y TN
        indices = np.arange(len(y_test))  # Índices de los ejemplos en el dataset

        tp_indices = indices[(y_test == 1) & (y_pred == 1)]
        tn_indices = indices[(y_test == 0) & (y_pred == 0)]
        fp_indices = indices[(y_test == 0) & (y_pred == 1)]
        fn_indices = indices[(y_test == 1) & (y_pred == 0)]

        if verbose:
            print(f"True Positives (TP) indices: {tp_indices.tolist()}")
            print(f"True Negatives (TN) indices: {tn_indices.tolist()}")
            print(f"False Positives (FP) indices: {fp_indices.tolist()}")
            print(f"False Negatives (FN) indices: {fn_indices.tolist()}")

        # Graficar matriz de confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión sobre el conjunto de prueba')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.show()

        return classification_report(y_test, y_pred), tp_indices, tn_indices, fp_indices, fn_indices


    def explore_bert_classifier_hyper_parameters(self, model_name='bert-base-uncased', n_trials=100, continue_searching=False, best_model_name = None, study_name = None, init_params = None, verbose=False, override=False):
        """
        Explore hyperparameters for a BERT classifier using Optuna with effective pruning.
        Optimized for disk usage by minimizing model checkpoints.
        """

        print(f"Exploring hyperparameters for {model_name} - using Optuna...")

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                item = {key: val.squeeze(0) for key, val in encoding.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)

        # Define paths for saving models and studies
        if best_model_name is None:
            best_model_path = os.path.join(self.output_dir, 'best_bert_model')
        else:
            best_model_path = os.path.join(self.output_dir, best_model_name)

        if study_name is None:
            study_path = os.path.join(best_model_path, "optuna_study.pkl")
        else:
            study_path = os.path.join(best_model_path, f"{study_name}.pkl")

        print(f"\n{'='*50}")
        print(f"BERT model path: {best_model_path} - Study: {study_path}")
        print(f"{'='*50}\n")

        try:
            if override:
                raise Exception("Override requested, starting fresh optimization")
            with open(os.path.join(self.output_dir, 'bert_model.pkl'), 'rb') as f:
                self.bert_model = pickle.load(f)
            print("Model loaded successfully")
            return self.bert_model
        except Exception as e:
            print(f"Starting new model optimization: {str(e)}")

            # Load dataset
            texts = self.data['text']
            labels = self.data['score']
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Prepare dataset
            train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer)
            test_dataset = TextDataset(X_test.tolist(), y_test.tolist(), tokenizer)

            # Create a custom callback for Optuna pruning
            class OptunaPruningCallback(TrainerCallback):
                def __init__(self, trial):
                    self.trial = trial
                    self.last_reported_epoch = -1

                def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                    if metrics is None:
                        return

                    current_epoch = int(state.epoch)  # Usamos el epoch como step
                    if current_epoch == self.last_reported_epoch:
                        return

                    # Reportamos la métrica usando el epoch como step
                    value = metrics.get("eval_f1", metrics.get("eval_accuracy"))
                    if value is not None:
                        self.trial.report(value, step=current_epoch)

                    self.last_reported_epoch = current_epoch

                    if self.trial.should_prune():
                        print(f"✂️ Pruning trial at epoch {current_epoch}")
                        raise optuna.exceptions.TrialPruned()

            # Create tracking variables to hold the best model and its metrics
            best_model_info = {
                'f1': -1.0,
                'trial_number': -1,
                'params': None,
                'model': None
            }

            def objective(trial):
                """Objective function to optimize hyperparameters"""

                try:
                    # Hyperparameters to optimize
                    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
                    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
                    dropout = trial.suggest_float("dropout", 0.1, 0.4)
                    weight_decay = trial.suggest_float("weight_decay", 1e-6, 0.1, log=True)
                    num_train_epochs = 15
                    warmup_steps = trial.suggest_int("warmup_steps", 0, 2000)
                    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 4, 8, 16, 32, 64, 128 ])
                    adam_beta1 = trial.suggest_float("adam_beta1", 0.75, 0.9999, log=True)
                    adam_beta2 = trial.suggest_float("adam_beta2", 0.85, 0.99999, log=True)
                    adam_epsilon = trial.suggest_float("adam_epsilon", 1e-9, 1e-6, log=True)

                    # Log trial information
                    print(f"\n{'='*50}")
                    print(f"Trial {trial.number}: Evaluating hyperparameters:")
                    for name, value in trial.params.items():
                        print(f"  {name}: {value}")
                    print(f"{'='*50}\n")

                    # Configure model with trial-specific dropout
                    config = AutoConfig.from_pretrained(
                        model_name,
                        num_labels=len(set(labels)),
                        hidden_dropout_prob=dropout,
                        attention_probs_dropout_prob=dropout
                    )

                    # Load model with adjusted config
                    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

                    # Use a temporary directory for this trial's outputs
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Set up training arguments - using temporary directory
                        training_args = TrainingArguments(
                            output_dir=tmp_dir,  # Use temporary directory for checkpoints
                            num_train_epochs=num_train_epochs,
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            evaluation_strategy="epoch",  # Evaluate after each epoch
                            save_strategy="epoch",
                            save_total_limit=1,  # Save only one model checkpoint
                            logging_dir=os.path.join(tmp_dir, "logs"),
                            logging_steps=50,
                            logging_first_step=True,
                            fp16=True,  # Enable mixed precision training
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            lr_scheduler_type="linear",
                            warmup_steps=warmup_steps,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            adam_beta1=adam_beta1,
                            adam_beta2=adam_beta2,
                            adam_epsilon=adam_epsilon,
                            report_to="none",
                            disable_tqdm=True,
                            load_best_model_at_end=True,
                            metric_for_best_model="eval_f1",
                            greater_is_better=True
                        )

                        # Set up metrics computation
                        def compute_metrics(pred):
                            labels = pred.label_ids
                            preds = np.argmax(pred.predictions, axis=1)
                            accuracy = accuracy_score(labels, preds)
                            f1 = f1_score(labels, preds, average="macro")
                            return {"accuracy": accuracy, "f1": f1}

                        # Create Trainer with callbacks
                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=test_dataset,
                            compute_metrics=compute_metrics,
                            callbacks=[OptunaPruningCallback(trial), EarlyStoppingCallback(early_stopping_patience=3)]  # Enable pruning
                        )

                        # Train the model
                        trainer.train()

                        # Evaluate the model
                        eval_result = trainer.evaluate()

                        # Log final results
                        print(f"\n{'='*50}")
                        print(f"Trial {trial.number} completed:")
                        for key, value in eval_result.items():
                            print(f"  {key}: {value:.4f}")
                        print(f"{'='*50}\n")

                        # Check if this is the best model so far
                        current_f1 = eval_result["eval_f1"]
                        if current_f1 > best_model_info['f1']:
                            print(f"New best model found in trial {trial.number}!")
                            # Update the tracking information
                            best_model_info['f1'] = current_f1
                            best_model_info['trial_number'] = trial.number
                            best_model_info['params'] = trial.params.copy()

                            # Save only the best model to disk
                            trainer.save_model(best_model_path)
                            tokenizer.save_pretrained(best_model_path)
                            print(f"Saved best model to {best_model_path}")

                    # Return the metric to optimize
                    return eval_result["eval_f1"]

                except optuna.exceptions.TrialPruned:
                    print(f"Trial {trial.number} was pruned.")
                    raise
                except Exception as e:
                    print(f"Error in trial {trial.number}: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    return float('-inf')  # Return a very low value for Optuna to discard

            # Try to load a previous study if continue_searching is True
            if continue_searching:
                try:
                    with open(study_path, 'rb') as f:
                        study = pickle.load(f)
                    print(f"Loaded existing Optuna study with {len(study.trials)} previous trials.")
                    print(f"Current best parameters: {study.best_params}")
                    print(f"Current best value: {study.best_value}")
                except FileNotFoundError:
                    print("No existing study found, starting a new one.")
                    study = optuna.create_study(
                        direction="maximize",
                        sampler=TPESampler(seed=42),
                        pruner=HyperbandPruner(
                            min_resource=1,  # Primer pruning después del 1er epoch
                            max_resource=15,  # Máximo de epochs (alineado con num_train_epochs fijo)
                            reduction_factor=3  # Agresividad del pruning)
                    ))

                    if init_params is not None:
                        study.enqueue_trial(init_params)

            else:
                print("Starting new optimization study.")

                study = optuna.create_study(
                    direction="maximize",
                    sampler=TPESampler(seed=42),
                    pruner=HyperbandPruner(
                        min_resource=1,  # Primer pruning después del 1er epoch
                        max_resource=15,  # Máximo de epochs (alineado con num_train_epochs fijo)
                        reduction_factor=3  # Agresividad del pruning)
                ))

                if init_params is not None:
                    study.enqueue_trial(init_params)

            # Run the optimization
            try:
                study.optimize(objective, n_trials=n_trials, timeout=None)
            except KeyboardInterrupt:
                print("Optimization interrupted. Saving current progress...")

            # Save the study
            with open(study_path, 'wb') as f:
                pickle.dump(study, f)

            # Print best trial information
            print("\n" + "="*80)
            print("Optimization finished.")
            print(f"Best trial: #{study.best_trial.number}")
            print(f"Best F1 score: {study.best_value:.4f}")
            print("\nBest parameters:")
            for param, value in study.best_params.items():
                print(f"  {param}: {value}")
            print("="*80 + "\n")

            # Try to visualize the results
            try:
                import matplotlib.pyplot as plt

                # Create figures directory
                figures_dir = os.path.join(self.output_dir, 'figures')
                os.makedirs(figures_dir, exist_ok=True)

                plt.figure()
                optuna.visualization.matplotlib.plot_optimization_history(study)
                plt.savefig(os.path.join(figures_dir, 'optimization_history.png'))

                plt.figure()
                optuna.visualization.matplotlib.plot_param_importances(study)
                plt.savefig(os.path.join(figures_dir, 'param_importances.png'))

                plt.figure()
                optuna.visualization.matplotlib.plot_slice(study)
                plt.savefig(os.path.join(figures_dir, 'param_slices.png'))

                print(f"Saved visualization plots to {figures_dir}")
            except Exception as e:
                print(f"Could not create visualizations: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

            return study.best_params

    def build_bert_classifier(self,
                              model_name="bert-base-uncased",
                              dropout=0.01,
                              num_train_epochs=6,
                              batch_size=32,
                              learning_rate=2e-5,
                              weight_decay=9.644,
                              warmup_steps=442,
                              gradient_accumulation_steps=2,
                              adam_beta1=0.9,
                              adam_beta2=0.999,
                              adam_epsilon=1e-8,
                              verbose=False,
                              override=False):

        """
        Create a BERT-based classifier to predict the score.
        """

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                item = {key: val.squeeze(0) for key, val in encoding.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

        try:
            if override:
                raise Exception("Override")
            with open(os.path.join(self.output_dir, 'bert_model.pkl'), 'rb') as f:
                self.bert_model = pickle.load(f)
            print("Model loaded successfully")
        except:

             # Load dataset
            texts = self.data['text']
            labels = self.data['score']
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load the config
            config = AutoConfig.from_pretrained(model_name, num_labels=len(set(labels)), hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)

            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config = config)

            # Prepare dataset
            train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer)
            test_dataset = TextDataset(X_test.tolist(), y_test.tolist(), tokenizer)

            training_args = TrainingArguments(
                    output_dir=self.output_dir,
                    num_train_epochs=num_train_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    eval_strategy="epoch",
                    save_strategy="no",
                    save_total_limit=0,
                    load_best_model_at_end=False,
                    logging_dir=os.path.join(self.output_dir, "logs"),
                    fp16=True,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    lr_scheduler_type="linear",
                    warmup_steps=warmup_steps,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    adam_beta1=adam_beta1,
                    adam_beta2=adam_beta2,
                    adam_epsilon=adam_epsilon,
                )

            trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    compute_metrics=lambda p: {
                        "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
                        "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="weighted"),
                    },
                )

            # Train the model
            trainer.train()

            # Evaluar el modelo
            predictions = trainer.predict(test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            model.save_pretrained(os.path.join(self.output_dir, f'model-{model_name}-{timestamp}'))
            tokenizer.save_pretrained(os.path.join(self.output_dir, f'tokeninzer-{model_name}-{timestamp}'))

            #Save the report as csv
            report = classification_report(y_test, y_pred, output_dict=True)

            hyperparams = {
                'model_name': model_name,
                'dropout': dropout,
                'num_train_epochs': num_train_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'warmup_steps': warmup_steps,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'adam_beta1': adam_beta1,
                'adam_beta2': adam_beta2,
                'adam_epsilon': adam_epsilon
            }

            self.single_output_model = model
            self.vectorizer_model = tokenizer
            self.generate_classification_report(y_test, y_pred, hyperparams, model_name)

            return {'model': model, 'tokenizer': tokenizer, 'report': report}

    def load_bert_model_from_path(self, model_path, tokenizer_path):
        return AutoModelForSequenceClassification.from_pretrained(model_path), AutoTokenizer.from_pretrained(tokenizer_path)

    def get_bert_classsifier_test_score(self, test_dataset_path, model_name='bert-base-uncased', best_model_path=None, timestamp=None, verbose=False):

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                item = {key: val.squeeze(0) for key, val in encoding.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item, idx  # Se retorna también el índice

        # Cargar el dataset
        test_dataframe = pd.read_csv(test_dataset_path)

        # Preprocesamiento
        self.preprocess_text_dataframe(test_dataframe, 'text', lower=True)

        # Cargar modelo y tokenizer
        if timestamp:
            model = AutoModelForSequenceClassification.from_pretrained(os.path.join(self.output_dir, f'model-{model_name}-{timestamp}'))
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.output_dir, f'tokeninzer-{model_name}-{timestamp}'))
        else:
            if best_model_path:
                model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
                tokenizer = AutoTokenizer.from_pretrained(best_model_path)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(os.path.join(self.output_dir, f'model-{model_name}'))
                tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.output_dir, f'tokeninzer-{model_name}'))

        self.single_output_model = model
        self.vectorizer_model = tokenizer

        # Mostrar los hiper parametros del modelo
        print(f"Model hyperparameters: {model.config}")

        # Crear dataset
        test_dataset = TextDataset(test_dataframe['text'].tolist(), test_dataframe['score'].tolist(), tokenizer)

        # DataLoader
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

        # Listas para predicciones, etiquetas y sus índices
        y_pred, y_test, indices = [], [], []

        model.eval()
        with torch.no_grad():
            for batch, batch_indices in test_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                predictions = torch.argmax(logits, dim=-1)

                y_pred.extend(predictions.cpu().numpy())
                y_test.extend(labels.cpu().numpy())
                indices.extend(batch_indices.numpy())

        # Calcular matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Identificar TP, FP, FN y TN
        tp_indices = [idx for idx, (true, pred) in zip(indices, zip(y_test, y_pred)) if true == pred and true == 1]
        tn_indices = [idx for idx, (true, pred) in zip(indices, zip(y_test, y_pred)) if true == pred and true == 0]
        fp_indices = [idx for idx, (true, pred) in zip(indices, zip(y_test, y_pred)) if true == 0 and pred == 1]
        fn_indices = [idx for idx, (true, pred) in zip(indices, zip(y_test, y_pred)) if true == 1 and pred == 0]

        if verbose:
            print(f"True Positives (TP) indices: {tp_indices}")
            print(f"True Negatives (TN) indices: {tn_indices}")
            print(f"False Positives (FP) indices: {fp_indices}")
            print(f"False Negatives (FN) indices: {fn_indices}")

        # Graficar matriz de confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión sobre el conjunto de prueba')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.show()

        return classification_report(y_test, y_pred), tp_indices, tn_indices, fp_indices, fn_indices

    def plot_dataframe_columns_count(self, columns):

        """
        Plot the columns of the dataframe
        @param columns: list of columns to plot
        """

        data = []

        for i in range(len(columns)):
            data.append(self.word_count_dataset[columns[i]].sum())

        print(f'Data: {data}')

        #sort the data
        data, columns = zip(*sorted(zip(data, columns)))

        cmap = cm.get_cmap('tab20c')  # You can change 'viridis' to any other colormap like 'plasma', 'inferno', etc.

        # Generate a color for each bar
        colors = cmap(np.linspace(0, 1, len(data)))

        #Plot the data
        fig, ax = plt.subplots()

        ax.bar(columns, data, color=colors)

        plt.show()

    def generate_classification_report(self, y_test, y_pred, hyperparams, model_name, output_dir=None):
        """
        Genera un reporte detallado de clasificación junto con los hiperparámetros usados
        y lo guarda en formato de texto y CSV.

        Args:
            y_test: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            hyperparams: Diccionario con los hiperparámetros usados
            model_name: Nombre del modelo utilizado
            output_dir: Directorio donde guardar los reportes

        Returns:
            Rutas a los archivos generados
        """

        if output_dir is None:
            output_dir = self.output_dir

        # Generar timestamp para los archivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Crear directorio si no existe
        save_dir = f"{output_dir}/{model_name}_report/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        # Calcular métricas
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Crear DataFrame para CSV (combinando métricas y hiperparámetros)
        df_report = pd.DataFrame(report_dict).transpose()

        # Crear un DataFrame para los hiperparámetros
        hyperparams_df = pd.DataFrame([hyperparams])

        # Guardar CSV con métricas
        csv_path = f"{save_dir}/report.csv"
        df_report.to_csv(csv_path)

        print(f"Classification report {report_dict}")

        # Guardar CSV con hiperparámetros
        hyperparams_csv_path = f"{save_dir}/hyperparams.csv"
        hyperparams_df.to_csv(hyperparams_csv_path, index=False)

        # Crear reporte de texto
        txt_path = f"{save_dir}/full_report.txt"

        with open(txt_path, 'w') as f:
            # Cabecera
            f.write("="*80 + "\n")
            f.write(f"REPORTE DE CLASIFICACIÓN: {model_name}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Métricas principales
            f.write("MÉTRICAS PRINCIPALES\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Score (Macro): {f1_macro:.4f}\n")
            f.write(f"F1 Score (Weighted): {f1_weighted:.4f}\n\n")

            # Reporte de clasificación detallado
            f.write("REPORTE DE CLASIFICACIÓN DETALLADO\n")
            f.write("-"*80 + "\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\n\n")

            # Hiperparámetros
            f.write("HIPERPARÁMETROS\n")
            f.write("-"*80 + "\n")
            for param, value in hyperparams.items():
                f.write(f"{param}: {value}\n")

        #Print the content of the report
        with open(txt_path, 'r') as f:
            print(f.read())

        # Generar y guardar matriz de confusión como imagen
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        conf_matrix_path = f"{save_dir}/confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(conf_matrix_path)
        plt.show()

        print(f"Reporte completo guardado en: {txt_path}")
        print(f"Métricas guardadas en: {csv_path}")
        print(f"Hiperparámetros guardados en: {hyperparams_csv_path}")
        print(f"Matriz de confusión guardada en: {conf_matrix_path}")

        return {
            'full_report': txt_path,
            'metrics_csv': csv_path,
            'hyperparams_csv': hyperparams_csv_path,
            'confusion_matrix': conf_matrix_path
        }


    def display_image(self, image_path, width=10, height=10):
        """
        Display an image given its path.

        Args:
        image_path (str): The path to the image file.
        """
        img = mpimg.imread(image_path)  # Read the image from the path
        plt.figure(figsize=(width, height))
        plt.imshow(img)  # Display the image
        plt.axis('off')  # Hide the axes
        plt.show()  # Show the plot