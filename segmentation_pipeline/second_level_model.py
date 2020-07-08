import os
import numpy as np
import openslide
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

parts = []
isup_grades = []

data = pd.read_csv('../train.csv')
counter = 0
for index, row in data.iterrows():
    if row['data_provider'] == 'karolinska':
        continue
    filename = '../train_label_masks/' + row['image_id'] + '_mask.tiff'
    if os.path.isfile(filename):
        isup_grades.append(int(row['isup_grade']))
        example = [0] * 4
        mask = openslide.OpenSlide(filename)
        total_pixels = mask.level_dimensions[2][0] * mask.level_dimensions[2][1]
        image = np.asarray(mask.read_region(location=(0, 0), level=2, size=mask.level_dimensions[2]))
        unique, counts = np.unique(image[:, :, 0], return_counts=True)
        d = dict(zip(unique, counts))
        for key, value in d.items():
            if key == 0:
                total_pixels -= value
            if key in [1, 2]:
                example[0] += value
            if key == 3:
                example[1] += value
            if key == 4:
                example[2] += value
            if key == 5:
                example[3] += value

        example = [i / total_pixels for i in example]
        parts.append(example)


X_train, X_test, y_train, y_test = train_test_split(parts, isup_grades, test_size=0.2, random_state=42)

clf = SGDClassifier(verbose=1)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print(confusion_matrix(y_test, preds))
