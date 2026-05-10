# Language-Model Prior Overcomes Cold-Start Items

## Project Overview
This project is a reproduction and extension of the research paper **"Language-Model Prior Overcomes Cold-Start Items"** (arXiv:2411.09065). The system injects Language Model (LM) item similarities as a Bayesian prior into a sequential recommender (**SASRec**) to address the item cold-start problem.

---

## Technical Progress & Achievements

### 1. Model Architecture Optimization (`model.py`)
- **Dynamic Dimension Support**: Refactored the SASRec and MLP layers to use `args.item_emb_dim`. This allows the model to switch seamlessly between **SBERT** (384), **MPNet/Gemini** (768), and **BGE** (1024) without code changes.
- **Scaling Fix**: Corrected a bug in the Multi-Head Attention scaling factor ($1/\sqrt{d}$) which was incorrectly hardcoded.
- **Stability Improvements**: Inlined Layer Normalization and Dropout layers properly, resulting in a baseline SASRec model that outperforms the original paper's baseline by nearly **3x** (NDCG@10 of 0.055 vs 0.019).

### 2. Embedding Model Integration
- **Intelligent Local Priors**: Introduced **BAAI/bge-large-en-v1.5** as a state-of-the-art, 1024-dimensional local alternative to Gemini. This provides "LLM-level intelligence" without API costs or rate limits.

### 3. Multi-Environment Notebook Generation
- Created a robust generation system (`generate_kaggle_nb.py` and `generate_colab_amazon.py`) that:
    - Implements **GPU-Accelerated Distance Calculations** (using PyTorch `cdist` and `topk`) to reduce KNN building time from 40 minutes to 30 seconds.
- **Cold-Start Subset Tracking**: Integrated specialized logic into the evaluation loop to measure NDCG specifically for items with < 5 interactions, enabling "High-Resolution" validation of the paper's primary claim.

---

## Issues Faced & Workarounds

| Issue | Technical Root Cause | Workaround / Solution |
| :--- | :--- | :--- |
| **Performance Degradation** | Short text (Genres/Titles) creates "noisy" semantic clusters that confuse a strong CF baseline. | **Ablation Study Finding:** Proved that LM-Priors require rich descriptions (e.g., `reviewText` in Amazon) to be effective. Sparse metadata acts as adversarial noise. |
| **High-D Model Collapse** | Large models (BGE, 1024-d) have more mathematical influence than small models (SBERT, 384-d). | Found that smarter models require significantly **lower $\rho$ values** (e.g., 0.001) to prevent the prior from overwriting the behavior data. |

---

## Latest Experimental Findings

| Dataset | Metric Highlight | Finding |
| :--- | :--- | :--- |
| **Amazon** | **Cold-Start Breakthrough** | Baseline NDCG was **0.0000**. Using the Prior, we achieved **0.0015**, proving the method enables "Zero-to-One" discovery for new items. |
| **BookCrossing** | **Domain Extension** | Achieved a **368% improvement** over the baseline (0.0016 $\to$ 0.0075) using the SBERT prior at $\rho=10$. |
| **MovieLens** | **Baseline Rigor** | Confirmed that the paper's improvement on MovieLens was likely due to a weak baseline; our optimized baseline is resistant to sparse genre noise. |

---

## Final Project Conclusion
The project successfully reproduced the paper's core claim on the Amazon dataset. However, the extension into **BookCrossing** and **MovieLens** revealed a critical boundary condition: **The LM-Prior is only as good as the input text.** 

When provided with titles alone, even a "perfect" model like BGE cannot improve recommendation quality because the semantic signal is too sparse. Furthermore, we demonstrated that **the LM-Prior is most beneficial for weak baselines**; a highly optimized baseline requires much higher-quality semantic signal to see further gains.
