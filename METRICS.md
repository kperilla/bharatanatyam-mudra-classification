# Metrics

## Methodology
A set of 9 Mudras were chosen (`['Sikharam', 'Tamarachudam', 'Sarpasirsha', 'Katakamukha_1', 'Tripathaka', 'Mukulam', 'Chandrakala', 'Suchi', 'Simhamukham']`),
and we measured the true positive classification rate for each. The average success rate is then made from the average of those
success rates.

For consistency, we have tried to put the camera in the same position, and rotate the hand in the following pattern:
- Do the gesture face-on with the camera,
- then slightly facing down,
- slightly facing up,
- facing left,
- facing right

One thing I've noticed is that the facing left and facing right are way more accurate

## Using the feature extraction + neural network
Sikharam: 59.2%,
Tamarachudam: 39.37%,
Sarpasirsha: 56.25%,
Katakamukha_1: 0%,
Tripathaka: 2.36%,
Mukulam: 9.76%,
Chandrakala: 32.56%,
Suchi: 33.06%,
Simhamukham: 20.63%

Average: 28.13%

## Adding the normalization step
Sikharam: 42.74193548387097
Tamarachudam: 61.41732283464567
Sarpasirsha: 77.95275590551181
Katakamukha_1: 3.1496062992125984
Tripathaka: 65.35433070866141
Mukulam: 42.74193548387097
Chandrakala: 40.15748031496063
Suchi: 58.19672131147541
Simhamukham: 39.37007874015748

Average: 47.898% (70% improvement!)

## Trying with a different gesture pattern

Here, I went from right facing to left facing, while slightly rotating the hand in each gesture:


{'label': 'Sikharam', 'valid_landmarks_ratio': 1.0, 'true_positive_classification_ratio': 0.5793650793650794}
{'label': 'Tamarachudam', 'valid_landmarks_ratio': 1.0, 'true_positive_classification_ratio': 0.9448818897637795}
{'label': 'Sarpasirsha', 'valid_landmarks_ratio': 1.0, 'true_positive_classification_ratio': 0.7795275590551181}
{'label': 'Katakamukha_1', 'valid_landmarks_ratio': 1.0, 'true_positive_classification_ratio': 0.016}
{'label': 'Tripathaka', 'valid_landmarks_ratio': 1.0, 'true_positive_classification_ratio': 0.7244094488188977}
{'label': 'Mukulam', 'valid_landmarks_ratio': 1.0, 'true_positive_classification_ratio': 0.3779527559055118}
{'label': 'Chandrakala', 'valid_landmarks_ratio': 1.0, 'true_positive_classification_ratio': 0.7952755905511811}
{'label': 'Suchi', 'valid_landmarks_ratio': 1.0, 'true_positive_classification_ratio': 0.656}
{'label': 'Simhamukham', 'valid_landmarks_ratio': 1.0, 'true_positive_classification_ratio': 0.44881889763779526}

Now that gets you 59.14%

I think you can make the argument, that no matter what you do, the head-on classification will always be inaccurate
