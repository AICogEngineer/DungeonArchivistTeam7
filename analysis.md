# Dungeon Archivist: Phase 3 Analysis Report

**Team Members:** Jared Chancey, David Barboza   
**Project:** The Dungeon Archivist - AI-Powered Asset Restoration

---

## Executive Summary

Our game studio acquired 3,917 dungeon game assets renamed to random hashes (e.g., a7f3d9.png) with no folder structure—a complete "data swamp." We built The Dungeon Archivist using three datasets: Dataset A trained a CNN to generate 256-dimensional embeddings stored in ChromaDB; Dataset B was automatically sorted using L2 distance metric and weighted voting; Dataset C (instructor-provided) will validate generalization during live demo.

**Key Achievement:** Understanding of the process of classifing unlabeled images using ChromaDB with L2 distance metric and weighted voting from CNN embeddings.

---

## 1. Problem Statement

### The Challenge
Our game studio acquired thousands of dungeon game assets (swords, monsters, walls, potions) in a completely disorganized state:
- All folder structure was lost
- Every file renamed to random hash (e.g., 9c21b5.png)
- 3,917 images in a single chaotic folder

### The Goal
Build an AI system to automatically restore order by:
1. Analyzing visual content of each image
2. Comparing against known examples
3. Sorting into proper categories

---

## 2. Technical Architecture

### Phase 1: Training the Vision Model
- **Dataset:** 6,031 labeled images from Dataset A
- **Model:** Keras CNN (Convolutional Neural Network)
- **Architecture:**
  - Input: 32×32×3 RGB images
  - Data augmentation: Random horizontal flip
  - 3 Convolutional blocks:
    - Block 1: Two Conv2D(32) layers + MaxPool + BatchNorm
    - Block 2: Two Conv2D(64) layers + MaxPool + SpatialDropout(0.3)
    - Block 3: Two Conv2D(128) layers + GlobalAveragePooling
  - Embedding layer: 256-dimensional feature vector (named "embedding_out")
  - Dropout: 0.5
  - Output: Softmax classification across 132 subcategories
- **Training:** Achieved 85% validation accuracy

### Phase 2: Vector Database Classification

#### How It Works

**Step 1: Generate Embeddings**
```
Chaos Image → CNN (up to embedding layer) → 256-D vector
```

**Step 2: Query Vector Database**
- All Dataset A images stored in ChromaDB with their 256-D embeddings
- For each chaos image, query for 5 most similar training images
- Similarity measured using L2 (Euclidean) distance metric

**Step 3: Weighted Voting**
```python
# Inverse distance weighting: closer neighbors vote stronger
weight = 1.0 / (distance + 1e-6)

# Example result:
Top 5 neighbors: [dungeon_wall, dungeon_wall, dungeon_doors, dungeon_wall, dungeon_floor]
Distances:       [42.5, 45.8, 48.2, 49.1, 52.3]

Weighted votes:
- dungeon_wall: weight = 1/42.5 + 1/45.8 + 1/49.1 = 0.0641
- dungeon_doors: weight = 1/48.2 = 0.0207
- dungeon_floor: weight = 1/52.3 = 0.0191

Winner: dungeon_wall → Copy to restored_archive/dungeon/wall/
```

**Step 4: Confidence Check & File Handling**
```
If nearest neighbor distance > 50.0 → Copy to review_pile/ (manual verification)
If nearest neighbor distance ≤ 50.0 → Auto-copy to appropriate category folder

Note: Files are COPIED (not moved) to preserve the original chaos folder
```

#### Threshold Strategy

We implemented a **conservative confidence threshold of 50.0** to prioritize classification precision over automation rate. This approach:
- Ensures high-confidence auto-sorting (only very similar images pass)
- Reduces risk of mislabeling
- Increases manual review volume (lower automation rate)
- Appropriate for production use where accuracy is critical

## 3. Why L2 Distance Metric?

### Distance Metric Comparison

| Metric | How It Works | Best For | Our Dataset? |
|--------|--------------|----------|--------------|
| **L2 (Euclidean)** | Straight-line distance in 128-D space | CNN image embeddings | **Selected** |
| Cosine | Angle between vectors | Text/word embeddings | Not Selected |
| Manhattan (L1) | Sum of absolute differences | Sparse data | Not Selected |

### Why We Chose L2

**Reason 1: CNN Optimization**
- Our CNN was trained using backpropagation, which naturally optimizes embeddings for Euclidean space
- During training, the model learns to place similar images close together in L2 distance

**Reason 2: Magnitude Matters**
- L2 considers both direction AND magnitude of feature vectors
- A sword with strong metallic features (high values) is different from a faded sword (low values)
- Cosine would ignore this distinction

**Reason 3: Performance**
- ChromaDB's HNSW (Hierarchical Navigable Small World) algorithm is optimized for L2
- Fast approximate nearest neighbor search: ~1-10ms per query

## 4. Results & Performance

### Phase 2 Classification Results

**Dataset B Processing:**
- Total images: 3,917
- Auto-sorted: 3,327 images 83%
- Sent to manual review: 590 images 17%

**Threshold Analysis:**
- Confidence threshold: 50.0

### Distance Observations

**Typical Distance Ranges (from our data):**
- Similar images: 0-50
- Different categories: 50-100+
- **Our confidence cutoff: 50.0**

**Interpretation:**
Our conservative threshold of 50.0 ensures only highly confident matches are auto-sorted. This prioritizes precision over automation rate, resulting in higher manual review volume but higher classification accuracy.

## 5. Technical Implementation Details

### Vector Database: ChromaDB

**Why ChromaDB?**
- Embedded vector database (no separate server needed)
- Built-in L2 distance support
- HNSW algorithm for fast similarity search
- Easy integration with Python



**Storage:**
- 6,029 embeddings from Dataset A
- Each embedding: 256 floats (1,024 bytes)
- Total database size: ~8-10 MB (including HNSW index)

### Batch Processing

**Efficiency Optimizations:**
- Process 32 images at once (batch processing)
- Single forward pass through CNN for entire batch
- Parallel Vector DB queries

---

## 6. Challenges

### Challenge 1: Small Dataset & Data Imbalance
**Problem:** The starting dataset was very small with some categories containing very little data to train and validate on.  
**Thoughts:** Merge parent folders to children folder with small amounts of data to train on. Even then, some datasets are overbearing (walls, floors, monsters) with large amounts of data compared to sparse categories (this will be a problem when learning to categorize later).

### Challenge 2: Balancing Embedding Dimension vs Confidence Threshold
**Problem:** Higher embedding dimensions allow the model to learn more data and recognize patterns, but also will require higher confidence threshold
because embeddings are more spread out across higher dimensions.   
**Thoughts:** Model training benefits (higher validation accuracy and lower validation loss) from larger embedding dimension, but high dimensions spread embeddings further in space. Set a threshold of 50.0 to balance auto-sorting at high dimensional space. Would be nice to follow up with a lower
dimensional embedding to see if we could push confidence threshold lower and keep auto-sorting percentage similar.

### Challenge 3: Datasets Vary In Art and Perspective
**Problem:** Between the three datasets, the art style and pespective of sprites varies wildly. This means additional data which our model has not trained on will have different patterns from the ones learned during training.   
**Thoughts:** Given the previous challenges, this is just an additional hurdle which will affect the final outcome. This is especially concerning given we will train a model from our categorization of dataset b to then categorize dataset c. This will result in the errors propagating and the models becoming less reliable the more data we train on given we expect the categorization process to be flawed.

### Challenge 4: Noise In Chaos Dataset From Backgrounds
**Problem:** Sprites in the chaos dataset had backgrounds which added additional noise that our model was not trained to handle.  
**Thoughts:** Use rembg library to remove background of chaos dataset sprites. This caused some of the sprites to be corrupted beyond recognition and led to them being categorized as walls or floors most of the time worsening the problem of data imbalance for categories. We were unable to use rembg on our training data because it would corrupt some data beyond recognition the gap between our model dataset and the chaos was also not solved (Note: using rembg on the chaos dataset did provide meaningful improvement in making embeddings closer allowing us to bring the confidence threshold from 5000 to 50 with similar results of auto-sorting percentage).

## 7. Key Findings

### What Worked Well  

**Vector Database Approach:** Successfully found images even for unlabeled data  
**Weighted Voting:** Robust against occasional mismatches in top-5 neighbors  
**Confidence Threshold:** Threshold for auto-sorting vs manual review worked as expected
**Batch Processing:** Efficient handling of 3,917 images in around a minute  

### Limitations  

**Distribution Shift:** Dataset B images differ from training data (high distances)  
**Manual Review Required:** 17% of images needed human verification  
**Category Imbalance:** Some rare categories had few training examples  
**Background Noise:** With Dataset B there was too much background noise which made the accuracy of sorting lower


## 8. Conclusions

### Did It Work?

**Technically Yes.** The Vector Database approach successfully:

1. **Classified unlabeled images** using only visual similarity
2. **Maintained high accuracy** through weighted voting and confidence thresholds
3. **Preserved detailed categorization** across many subcategories
4. **Processed efficiently** using batch operations and fast vector search

### Why This Approach Works

**The Key Insight:**
Our CNN learned to compress 32×32×3 (3,072 numbers) images into 256-dimensional embeddings where similar images cluster together. The Vector Database simply finds which cluster each new image belongs to.

**Mathematics Behind It:**
- CNN embedding layer learns: image → 256-D vector
- Similar images: Similar vectors (low L2 distance)
- Vector DB finds: Closest distances between vectors

---

## 9. Phase 3: Expansion Analysis

### Retraining with Combined Dataset

**Before (Original Model):**
- Training data: 6,031 images (Dataset A only)
- Validation accuracy: 85%

**After (Retrained Model):**
- Training data: 9,358 images (Dataset A + verified Dataset B)
- Validation accuracy: 80%

**Conclusion:**
Did adding Dataset B improve performance? No because we ran into problems with the background causing noise on dataset b and that affected the accuracy of the model

## Final Conclusion

The Dungeon Archivist successfully restored structure to 3,917 previously chaotic game assets by combining CNN-based visual embeddings with a vector database–driven classification pipeline. Through Phase 1 and Phase 2, the system demonstrated that image similarity in embedding space, measured via L2 distance and reinforced through weighted voting, is an effective strategy for sorting unlabeled visual data at scale.

In Phase 2, the system achieved:

83% automatic classification with high confidence

17% manual review, routed conservatively to prevent mislabeling

Preservation of subcategories

Efficient processing via batch inference and fast nearest-neighbor search

Phase 3 provided critical insight into the system’s limitations. Retraining the CNN using a combined dataset (Dataset A + verified Dataset B) resulted in a 5% drop in validation accuracy (85% → 80%). This confirmed that simply increasing data volume does not guarantee better performance. The primary cause was background noise and visual inconsistency in Dataset B, which introduced feature interference and reduced the model’s ability to learn clean, discriminative embeddings.
