import os
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser

MODEL_DIR_PATH = 'model_test'
DATASET_PATH = 'datasets'

def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(f'[Training] {img_path}')
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels


def load_data(dataset_path, features_extractor):
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    dataset = datasets.ImageFolder(dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    return embeddings, labels, dataset.class_to_idx


def train(embeddings, labels):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    return softmax.fit(embeddings, labels)


def startTraining():
    features_extractor = FaceFeaturesExtractor()
    embeddings, labels, class_to_idx = load_data(DATASET_PATH, features_extractor)
    clf = train(embeddings, labels)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))

    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join('model', 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)


if __name__ == '__main__':
    startTraining()
