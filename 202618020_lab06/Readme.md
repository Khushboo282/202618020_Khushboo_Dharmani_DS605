# DS605 Lab 6 — Image & Text Feature Extraction with Classical ML

Turning raw images and text into numeric features, then classifying them with traditional scikit-learn models 

- **Part A:** crack / non-crack asphalt images → hand-made intensity + Canny edge features → Logistic Regression, SVM, Random Forest
- **Part B:** spam / non-spam emails → `CountVectorizer` vs `TfidfVectorizer` → Naive Bayes, Logistic Regression
- **Part C:** improving the text representation (stop words, vocabulary size, `min_df`, bigrams) and discussing the trade-offs

## Repository layout

```
.
├── analysis.ipynb        # full runnable notebook (Parts A, B, C + summary)
├── requirements.txt
├── README.md
├── data/
│   ├── asphalt/          # Asphalt Crack Dataset (Mendeley Data), 400 images
│   └── spam/emails.csv   # Email Spam Classification Dataset (Kaggle), 5,172 rows
└── results/
    ├── sample_images.png              # colour vs grayscale samples
    ├── canny_edges.png                # grayscale vs Canny edges samples
    ├── image_features.csv             # one feature row per image + label
    ├── image_results.csv              # Part A model metrics
    ├── image_confusion_matrices.png
    ├── class_distribution.png         # spam vs non-spam counts
    ├── text_results.csv               # Count vs TF-IDF comparison
    ├── text_confusion_matrices.png
    ├── text_improvement_results.csv   # Part C comparison
    └── improvement_comparison.png
```
## Image features

**Pipeline:** read with OpenCV → check shape/dtype (448×448×3, uint8) → resize to 128×128 → grayscale → features → stratified 80/20 split → models.

**Features per image (12 + label):**

| Group | Features |
|---|---|
| Intensity (NumPy) | mean brightness, contrast (std), min, max, median, 10th/90th percentile, dark-pixel ratio (< 60), bright-pixel ratio (> 200), histogram entropy |
| Edges (OpenCV) | Canny edge count, edge density (Gaussian blur 5×5, thresholds 100/200) |

**What separates the classes (class means):**

| Feature | Non-crack | Crack |
|---|---|---|
| edge_density | 0.003 | 0.052 |
| edge_count | 50 | 856 |
| dark_ratio | 0.002 | 0.016 |
| min brightness | 68 | 30 |
| contrast_std | 24.4 | 31.3 |
| mean brightness | 157 | 144 |

**Results (80 test images):**

| Model | Accuracy | Precision | Recall | F1 | Train (s) | Predict (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.925 | 0.947 | 0.900 | 0.923 | 0.041 | 0.0012 |
| SVM (RBF) | 0.950 | 0.950 | 0.950 | 0.950 | 0.010 | 0.0014 |
| Random Forest | 0.950 | 0.929 | 0.975 | 0.951 | 0.189 | 0.0043 |

Random Forest is preferred because recall matters most (a missed crack is the costly error): it found 39 of 40 cracks. With only 80 test images, a one-image difference is 1.25%, so the model ranking is not statistically strong.

## Text vectorization

**Dataset note:** `emails.csv` is already a bag-of-words table (3,000 word-count columns + `Prediction`), not raw text. Each row is rebuilt into a pseudo-document by repeating each word by its count, so **word order is not preserved**. After removing empty and duplicate rows: 4,631 emails (3,170 non-spam / 1,461 spam).

Cleaning: lowercase, remove `subject:`, replace URLs, keep letters only, collapse whitespace. Vectorizers are fit on the training split only.

**Count vs TF-IDF** (copy the numbers from `results/text_results.csv` after running):

| Vectorizer | Model | Features | Accuracy | F1 | Vectorize (s) | Train (s) | Predict (s) |
|---|---|---|---|---|---|---|---|
| CountVectorizer | MultinomialNB | | | | | | |
| CountVectorizer | LogisticRegression | | | | | | |
| TF-IDF | MultinomialNB | | | | | | |
| TF-IDF | LogisticRegression | | | | | | |

The vocabulary is 2,974 words rather than 3,000 because the default tokenizer drops one-letter tokens.

## Improving the representation (text)

TF-IDF + Logistic Regression, one change per row:

| Config | Features | Accuracy | F1 | Vectorize (s) | Train (s) |
|---|---|---|---|---|---|
| baseline (all words) | 2,974 | 0.972 | 0.955 | 1.14 | 0.034 |
| stop words removed | 2,738 | 0.964 | 0.943 | 1.02 | 0.024 |
| bigrams, max_features=10000 | 10,000 | 0.962 | 0.939 | 1.70 | 0.502 |
| max_features=1000 / 500, min_df=50 | | | | | |


**Observations**
- Removing stop words and capping features gave no accuracy gain here; the baseline was best.
- Bigrams quadrupled the feature count, slowed training about 15×, and did not help. They are also not meaningful on this data because word order was lost when the text was rebuilt.
- A much smaller vocabulary is where the real computation saving is; compare its accuracy drop against the baseline in the last row.
- Trade-off: more features → more time, not necessarily more accuracy; fewer features → faster, small accuracy cost.

## Constraints followed

No CNNs, deep learning or pretrained embeddings. Image features come only from NumPy/OpenCV; scikit-learn is used for vectorizers and classical models.