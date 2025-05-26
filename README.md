This project focused on binary news source classification using only article headlines, specifically distinguishing between Fox News and NBC.

## Summary

The primary goal was to develop machine learning models that could identify subtle linguistic cues in headlines to accurately predict their source.  The dataset comprised approximately 5,000 headlines, collected through web scraping and augmented to ensure class balance and diversity. A comprehensive preprocessing pipeline was implemented, including lowercasing, removal of standard and custom stopwords, and headline cleaning. The custom stopword list was designed to eliminate branding-related or layout-specific terms (e.g., "fox", "nbe", "breaking") to prevent label leakage.

Various classical models such as Logistic Regression, Support Vector Machines, Multinomial Naive Bayes, Random Forests, and XGBoost were tested. These were coupled with multiple feature extraction strategies, including TF-IDF, Count Vectorizer, and Hashing Vectorizer. An experiment with semantic embeddings using Sentence-BERT (MiniLM) was also conducted, but it underperformed simpler TF-IDF approaches, likely due to the short length and limited context of headlines. 

Each pipeline underwent systematic optimization using GridSearchCV with cross-validation and was evaluated using consistent metrics such as accuracy, F1-score, and ROC AUC. The best-performing model was a Support Vector Machine with an RBF kernel based on TF-IDF bigrams, achieving approximately 80% accuracy. This model surpassed all baselines. Exploratory experiments on n-gram patterns, vectorizer types, and semantic embeddings further validated the design choices. The project concluded that classical models with rich lexical features like bigram TF-IDF are highly effective for short-text classification tasks, often outperforming deeper semantic models on practical metrics.

## Core Components

### 2.1 Data Collection and Cleaning

#### Procedure and Protocol for Data Collection

The initial dataset included 3,805 article URLs from Fox News and NBC. To improve generalizability and class balance, an additional 700 NBC and 500 Fox News headlines were manually scraped using `requests` and `BeautifulSoup`. Headlines were extracted by parsing outlet-specific HTML structures. For instance, Fox News headlines were typically found within an `<h1>` tag with the class `headline speakable`, while NBC's layout required multiple fallback strategies. To ensure robustness, HTTP status codes were verified, duplicate headlines were removed, and pages with scraping failures were filtered out. The final raw dataset contained 5,200 headlines, evenly balanced between the two outlets. 

#### Curation and Cleaning

The scraped headlines were then cleaned using a structured NLP pipeline:

  * **Lowercasing**: All text was converted to lowercase for normalization.
  * **Stopword Strategy**:
      * **Default Stopword Removal**: Standard English stopwords (e.g., "the", "is", "and") were removed using `nltk.corpus.stopwords` as they add syntactic structure but little semantic value. 
      * **Custom Stopword List**: A custom list was built to remove non-informative words strongly correlated with outlet branding, layout cues, or visual presentation rather than actual content. These included direct mentions of the outlet (e.g., 'fox', 'nbe') and formatting or callout words (e.g., "news", "update", "report"). This list was saved to `custom_stopwords.txt` for consistent application across all vectorizers and modeling pipelines.
  * **Character Sanitization**: HTML tags, escape characters, and extra spaces were removed. Stemming/lemmatization was also applied in some cases to compare performance.

The dataset was stored with columns for title, outlet (Fox or NBC), label (Fox or NBC), and URL

### 2.2 Model Design

#### 2.2.1 Initial Design Considerations

A baseline model was established using Logistic Regression with TF-IDF vectorized unigrams, chosen for its interpretability, low complexity, and proven performance in short-text classification. This baseline yielded an initial test accuracy of approximately 66%, confirming that a linear model could identify surface-level linguistic patterns. However, it was noted that unigram representation might be too shallow, and the model capacity too limited, to capture editorial nuances.

#### 2.2.2 Feature Representation Experiments

Various text vectorization strategies were systematically explored:

  * **TF-IDF Vectorizer**: This was the primary representation, tuned with different `max_features` (1000 to 5000) and `ngram_range` values. Adding bigrams improved performance by capturing editorial phrasings. 
  * **Count Vectorizer**: Primarily used with Naive Bayes and SVM, where raw term frequency often performs as well as TF-IDF.
  * **Hashing Vectorizer**: Explored for efficiency and fixed memory footprint, but not used in final models due to lack of interpretability.
  * **Sentence Embeddings (MiniLM-BERT)**: `all-MiniLM-L6-v2` embeddings were used to encode headlines as dense, semantic vectors. These were evaluated with Logistic Regression and Random Forest but consistently lagged behind TF-IDF models, likely due to the short, sparse nature of headlines providing limited context.

#### 2.2.3 Modeling and Optimization Process

Each model was paired with its respective feature representation within a Pipeline and tuned using GridSearchCV with k-fold cross-validation. Key models explored:

  * **Logistic Regression**: A consistent performer and the baseline. Tuned via `C`, `solver`, and `max_iter`. Bigrams and increased feature size led to substantial performance gains.
  * **Multinomial Naive Bayes**: Paired with Count Vectorizer and TF-IDF. It was simple, fast, interpretable, and effective with balanced class distributions. Also used to extract predictive tokens per class. 
  * **Support Vector Machines (SVM)**: Both linear and RBF kernels were explored. RBF kernels yielded better ROC-AUC scores but had longer training times. Bigram TF-IDF with RBF kernel offered near-best overall performance.
  * **Random Forest & XGBoost**: Tree-based models were effective with TF-IDF. Parameters like `max_depth`, `n_estimators`, and `learning_rate` (for XGBoost) were tuned. While these models performed reasonably well (around 76% accuracy), they did not surpass SVM's performance and were not selected as the final model due to longer training times and less consistent generalization. 

### 2.3 Evaluation Protocol and Results

A rigorous evaluation strategy was employed, based on multiple performance metrics and controlled cross-validation.

  * **Train-Test Split**: An 80-20 stratified split ensured class balance between Fox News and NBC headlines in both training and test sets. All model training and tuning were performed exclusively on the training set to avoid data leakage.
  * **Cross-Validation**: K-fold cross-validation within GridSearchCV was used to tune hyperparameters, ensuring model robustness to overfitting. 
  * **Metrics**: Models were evaluated using Accuracy (overall correctness), F1-score (balance precision and recall), and ROC AUC (ability to separate classes across thresholds). 
  * **Model Selection**: The final model was selected based on a combination of test accuracy, F1-score, and ROC AUC.
      * Logistic Regression with TF-IDF unigrams achieved 66% accuracy as a baseline. 
      * Adding `ngram_range = (1,2)` improved performance to 77% accuracy with better F1-scores. 
      * SVM with TF-IDF bigrams reached 78-79% accuracy, though it was slower to train. 
      * Random Forest and XGBoost with TF-IDF bigrams achieved approximately 76% test accuracy and ROC AUC nearing 0.85. 

All evaluations were conducted on the held-out test set to simulate real-world performance. Confusion matrices and ROC curves were used for visualization and to investigate misclassification patterns, guiding final model selection and exploratory investigations. On an instructor-provided 20-sample mini test set, models achieved 70% accuracy with Random Forest, 85% with Naive Bayes, and 90% with SVM (all using TF-IDF features). Based on these results, SVM was chosen as the final model for full test inference. 

### Table 1: Performance Comparison Across Models and Embedding Techniques 

| Model | Embedding | Key Parameters | Acc | F1 | ROC AUC |
| :-------------------- | :------------------------------ | :------------------------------------------------- | :---- | :---- | :------ |
| Logistic Regression | TF-IDF (Unigram) -Baseline | max\_features = 100, C=1.0 | 0.66 | 0.67 | 0.74 |
| | TF-IDF (Unigram + Bigram) | features=5000, C=1.0 | 0.77 | 0.78 | 0.86 |
| | Sentence-BERT (MiniLM) | 384-d vector, solver: saga | 0.734 | 0.74 | 0.81 |
| Naive Bayes | TF-IDF (Unigram + Bigram) | max\_features = 5000 alpha=1.0 | 0.774 | 0.77 | 0.854 |
| | Count Vectorizer (Unigram + Bigram) | max features=5000, alpha=1.0 | 0.775 | 0.77 | 0.86 |
| | Hashing Vectorizer | max\_features=5000, alpha=1.0 | 0.736 | 0.77 | 0.81 |
| SVM | TF-IDF (Unigram + Bigram) | C=10 kernel=rbf, gamma scale | 0.78 | 0.77 | 0.863 |
| | Count Vectorizer (Unigram + Bigram) | kernel rbf, C=10, gamma scale | 0.786 | 0.79 | 0.864 |
| | Hashing Vectorizer | C=10 kernel=rbf, gamma scale | 0.76 | 0.76 | 0.84 |
| Random Forest | TF-IDF | Default Params | 0.76 | 0.76 | 0.842 |
| | TF-IDF (Unigram + Bigram) | n\_estimators = 200, max\_depth = 50, min\_samples\_split = 5, max features = log2 | 0.76 | 0.75 | 0.843 |
| | BERT GridSearch | n\_estimators = 250, max\_depth = 40, max features = sqrt | 0.66 | 0.66 | 0.72 |
| XGBoost | TF-IDF (Unigram) | max\_features = 4000, max\_depth = 5 lr=0.1 | 0.71 | 0.72 | 0.80 |
| | Count Vectorizer (Bigram) | max\_features = 3000, nestimators = 200, max\_depth = 5, lr=0.1 | 0.732 | 0.73 | 0.81 |
| | Hashing Vectorizer | n\_estimators = 20, max.depth=5, lr=0.1 | 0.71 | 0.72 | 0.879 |

## 3 Exploratory Questions

### 3.1 Which vectorizer performs best: TF-IDF, Count Vectorizer, Hashing Vectorizer, or Sentence-BERT? 

**Question and Motivation**: We aimed to evaluate which feature extraction technique best represents short headlines for source classification. Specifically, we compared Count Vectorizer, TF-IDF, HashingVectorizer, and semantic embeddings driven by Sentence-BERT (MiniLM) to determine their effectiveness across multiple classifiers. 
**Prior Work / Expectations**: From course material and research, we expected TF-IDF or Sentence-BERT to outperform Count Vectorizer due to their ability to capture either term rarity or semantic meaning. However, some literature suggests that for short texts, simpler frequency-based encodings like Count Vectorizer may be sufficient or even superior due to limited context windows. 
**Methods**: We trained models (Logistic Regression, Naive Bayes, SVM, and Random Forest) on different vector representations of the same dataset. We tuned each model using GridSearchCV, and evaluated on held-out test sets using Accuracy and ROC AUC. 
**Results and Updated Beliefs**: Contrary to our expectations, TF-IDF consistently outperformed Count Vectorizer across all models, especially when bigrams were used. Sentence-BERT underperformed TF-IDF, likely due to the extremely short length of headlines limiting its ability to model contextual semantics. Hashing Vectorizer worked acceptably in tree-based models but lacked interpretability. 
**Limitations**: We did not explore Word2Vec due to limited dataset size and headline length. Also, Sentence-BERT embeddings were not fine-tuned, which may explain their weaker performance. Additional experiments with contextual finetuning or longer text inputs may shift these results. 

### 3.2 Do bigrams significantly improve classification performance? 

**Question and Motivation**: We hypothesized that editorial framing differences between outlets would appear in multi-word phrases (e.g., "biden admin", "school shooting"). Hence, we explored whether including bigrams would improve performance over unigrams alone. 
**Prior Work / Expectations**: N-gram models are widely used in traditional NLP, and it's known that bigrams capture more semantic content. Prior results in short-text classification support the idea that bigrams offer strong returns on performance in sparse contexts. 
**Methods**: While we isolated unigrams vs bigrams in controlled experiments only for Logistic Regression, we incorporated both ngram range $=(1,1)$ and (1,2) into our GridSearchCV pipelines across all models (Logistic Regression, SVM, Naive Bayes, and XGBoost). This allowed the models to select between unigrams and bigrams based on cross-validation performance. 
**Results and Updated Beliefs**: In nearly all cases, the best-performing configurations used ngram range $=(1,2)$, suggesting that bigrams provided useful signal across models. For example, our Logistic Regression model improved from a baseline of 66% (with default unigrams) to 77% with tuned parameters that included bigrams.  SVM and XGBoost models also selected bigram-based TF-IDF as part of their optimal configuration, leading to improvements of 2-4% in both Accuracy and ROC AUC. 
**Limitations**: Bigrams increase feature dimensionality and sparsity, which may hurt performance if the dataset were smaller or more imbalanced. We did not explore trigrams due to runtime concerns, though they may offer marginal gains. 

## 4 Team Contributions

  * Ashay Katre: Led SVM, Logistic Regression, and Naive Bayes Classifier modeling. Designed exploratory experiments, feature comparison framework, and managed report writing and refinement. 
  * Vinod Ghanchi: Handled data augmentation by developing web scraping pipeline, and creating custom stopwords. Focused on Random Forest modeling and hyperparameter tuning. Assisted with preparing visualizations. 
  * Shashank Kambhatla: Worked on XGBoost Classifier, and contributed to EDA and exploratory question evaluations, as well as preparation of slides. 

## Conclusion

TF-IDF with bigrams consistently outperformed other embeddings (including Sentence-BERT), proving that classic lexical features remain highly effective for short headline classification. Support Vector Machines (SVM) with RBF kernel achieved the best overall performance with 80% accuracy and ROC of 0.86. Feature design and preprocessing (e.g., custom stopwords) had as much impact as model complexity, reinforcing the importance of data-centric approaches in applied ML tasks. 

## Appendix

### A. Baseline Simple Logistic Regression
![image](https://github.com/user-attachments/assets/ab8c7804-1827-44ec-bb74-828d9c6dc548)


![image](https://github.com/user-attachments/assets/5c49c146-2415-4231-b3bc-0a9c4aaa49bf)


### B. Naive Bayes

![image](https://github.com/user-attachments/assets/13e5f098-6768-487f-ad7a-d22447b3433b)


![image](https://github.com/user-attachments/assets/f1686272-898c-4de7-b1d6-0d3836bd21f8)


![image](https://github.com/user-attachments/assets/f69d6ec4-3d71-4eaf-bafa-105a95d8975f)


### C. SVM

![image](https://github.com/user-attachments/assets/997c6517-d56d-4938-9679-04ff57add75b)


![image](https://github.com/user-attachments/assets/1dd08960-9bc9-430b-b179-3d76e68e751b)


![image](https://github.com/user-attachments/assets/cd551cbc-8b2d-478c-8149-9cb8ae1d519c)


![image](https://github.com/user-attachments/assets/1b685086-c96f-4ff3-b0e5-040c132cdb86)


### D. Random Forest

![image](https://github.com/user-attachments/assets/e2b1041a-261d-4cd0-8e4e-a464ba311709)


![image](https://github.com/user-attachments/assets/ecee6cd0-6480-46b6-8cad-a8f5e7d95ec3)


![image](https://github.com/user-attachments/assets/acae2275-f791-4262-b4ee-afedcb8597e3)


![image](https://github.com/user-attachments/assets/fcf82d07-6bc5-467c-bb01-93e1c808e7be)

### E. XGBoost

![image](https://github.com/user-attachments/assets/f4ff33f9-3c4e-4869-8a03-ab9a6d85cb25)


![image](https://github.com/user-attachments/assets/4f31d39c-946b-4354-bfaf-0d86e2fa11b4)


![image](https://github.com/user-attachments/assets/cc0f68a9-f518-4f76-90a6-85028cb6759e)
