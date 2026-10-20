# Image Representation, PCA, Eigenfaces and Face Recognition

DS605 Fundamentals of Machine Learning, Lab Assignment 7
Dataset: AT&T / ORL Database of Faces (40 subjects x 10 images, 112 x 92 grayscale)

## Approach

Images -> grayscale pixel matrix -> flatten -> mean centering -> PCA via eigendecomposition and SVD (NumPy only) -> explained variance -> eigenfaces -> reduced representation -> reconstruction -> nearest-match recognition -> traditional ML classifiers.

**Part A: representation and split.** Each image is a 112 x 92 matrix with values 0 to 251. Flattened, it becomes a 10,304-dimensional vector, giving a 400 x 10,304 matrix. The split is stratified and seeded (seed 42): 8 train and 2 test images per subject (320 / 80), so every subject appears in both sets. The split is written in NumPy.

**Part B: PCA.** PCA is implemented with NumPy linear algebra only (no scikit-learn PCA).
- The training mean face is subtracted from every image. Test images use the *training* mean.
- Method 1: eigendecomposition of the small 320 x 320 matrix `Xc Xc^T / (n-1)`, mapped back to pixel space (`Xc^T v`). This avoids building the 10,304 x 10,304 covariance matrix and is the classic Turk and Pentland eigenfaces trick.
- Method 2: SVD of the centered matrix, with eigenvalues `S^2 / (n-1)`.
- The two methods agree: eigenvalues match, and the top 100 directions have |cosine| of 1.0 (to about 1e-14), up to sign. Centered data has rank at most n-1, so 319 components are kept.

## Results

### Explained variance (`explained_variance.png`)

| Variance kept | Components |
|---|---|
| 80% | 42 |
| 90% | **99** |
| 95% | 164 |
| 99% | 267 |

k = 99 is the minimum number of components that preserves at least 90% of the variance (actual: 90.01%).

### Eigenfaces (`eigenfaces.png`)

Top 16 eigenfaces, with the variance each explains.
TODO: describe what you see (e.g. which components look like lighting patterns, which look like facial features such as eyes, glasses or mouth).

### Reconstruction and storage (`reconstructions.png`, `reconstruction_error.png`)

| | MSE | PSNR |
|---|---|---|
| Train, k = 99 | 154.13 | 26.25 dB |
| Test, k = 99 | 323.21 | 23.04 dB |

- The training error matches the theoretical value (sum of discarded eigenvalues x (n-1)/n) exactly: 1,588,148.4 per image.
- Test error is about twice the train error, because test faces were not used to learn the directions.

Storage (training set, 320 faces):

| Representation | Size |
|---|---|
| Original pixels, uint8 | 3.30 MB (10,304 bytes per face) |
| Original pixels, float64 (in memory) | 26.4 MB |
| PCA codes, float64 | 253 KB (792 bytes per face) |
| PCA model (99 eigenfaces + mean face), float64 | 8.24 MB |
| PCA codes + model | 8.50 MB |

Each face needs 104x fewer numbers, but the stored basis is larger than the 8-bit training images. PCA saves storage only when the basis is shared across many more faces than components (or when many more faces are stored).

### Recognition by nearest match (`accuracy_vs_k.png`, `matches.png`)

Each test image is matched to its nearest training image by Euclidean distance.

| Representation | Test accuracy |
|---|---|
| Raw pixels (10,304 dims) | 98.75% (79/80) |
| PCA, k = 99 | 98.75% (79/80) |

- Keeping all 319 components gives predictions identical to raw pixels (verified in the notebook), so any difference comes only from discarded components.
- Accuracy vs components: k = 5 gives 90%, k = 10 gives 97.5%, and from k = 30 onward it stays at 98.75%. The 90%-variance rule (k = 99) is more than this task needs.
- Only one test image is misclassified: test index 62, true subject 32, matched to subject 2.
  TODO: describe what differs between the two faces in `matches.png` (lighting, expression, glasses, pose).

Robustness over 20 random splits (PCA refit on each split's training data):

| Split | Raw pixels | PCA k = 20 | PCA k = 50 | PCA k = 99 |
|---|---|---|---|---|
| 8 train / 2 test | 97.75% +/- 1.92 | 97.56% +/- 1.83 | 98.00% +/- 1.74 | 97.94% +/- 1.87 |
| 5 train / 5 test | 93.87% +/- 2.07 | 92.20% +/- 1.55 | 93.67% +/- 1.97 | 93.40% +/- 2.21 |

PCA with 50 to 99 components performs the same as raw pixels within noise. Too few components (k = 20) costs about 1.5 points when training data is scarce. Going from 8 to 5 training images per person costs about 4 points for every method.

### Traditional ML on PCA features (`ml_comparison.csv`, `ml_cv_vs_k.png`)

Number of components and model settings were chosen by 4-fold cross-validation on the training set only (PCA refit inside each fold). The test set was used once, at the end.

| Method | Settings | Components | CV accuracy | Test accuracy |
|---|---|---|---|---|
| SVM (RBF) | C = 10 | 99 | 98.13% | 98.75% |
| KNN | 1 neighbor | 30 | 97.19% | 98.75% |
| Nearest match (PCA) | Euclidean | 99 | n/a | 98.75% |
| Nearest match (raw) | Euclidean | 10,304 | n/a | 98.75% |
| Logistic Regression | C = 1 | 50 | 96.88% | 96.25% |
| SVM (linear) | C = 0.1 | 75 | 96.56% | 95.00% |

## Observations

- PCA compresses each face from 10,304 numbers to 30 to 99 with no loss in recognition accuracy.
- SVM-RBF has the best cross-validation score, but it ties the simple nearest-match baselines on the test set. One test image is 1.25%, so the test differences between the top four methods are within noise.
- The best KNN uses 1 neighbor, which makes it the same method as the nearest-match baseline. With only 8 images per person, a larger neighborhood includes other people's faces.
- The linear models score lower. TODO (optional): state whether removing `StandardScaler` changed this.
- Reconstruction quality drops on unseen faces (test MSE about 2x train MSE).

