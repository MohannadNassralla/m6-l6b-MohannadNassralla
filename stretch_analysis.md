# Embedding Space Analysis

## 1. Methodology
For this visualization, I applied **t-SNE** to the word-level GloVe vectors and **PCA** to the document-level DistilBERT embeddings. 
- **t-SNE** was chosen for words because it excels at preserving local neighborhoods, allowing us to see if "synonyms" or "category-mates" cluster tightly. 
- **PCA** was used for documents to capture the maximum variance in the dataset, which effectively separates broad topics like "Sport" from "Tech" across the primary axes.

## 2. Chosen Categories
I selected five distinct categories for word embeddings:
1.  **Countries**: Geographic entities.
2.  **Sports**: Action and competition terms.
3.  **Finance**: Economic and corporate terminology.
4.  **Technology**: Digital and computational terms.
5.  **Emotions**: Subjective human states.

These categories were chosen to test the model's ability to differentiate between concrete nouns (Countries), abstract states (Emotions), and domain-specific jargon (Finance/Tech).

## 3. Visual Interpretation
The embedding plots reveal several key insights into how models organize human language:

*   **Semantic Clustering:** In the GloVe plot, the "Emotions" category forms a distinct cluster far from "Finance," showing that the model successfully captures the difference between sentiment-heavy language and objective business language.
*   **Topic Separation in Documents:** The BBC News PCA plot shows that "Sport" and "Tech" articles are at opposite ends of the space. This is expected as their vocabularies share very little overlap.
*   **Outliers and Overlap:** Interestingly, some overlap occurs between "Business" and "Politics" in the document space. This reflects real-world model behavior where a political article regarding trade tariffs contains high amounts of economic terminology, causing the DistilBERT vectors to converge in high-dimensional space.
*   **Dimensionality Reduction Trade-offs:** While t-SNE produced tighter, more "human-readable" word clusters, the PCA plot for documents provided a clearer "map" of how distinct the BBC sections are from one another.