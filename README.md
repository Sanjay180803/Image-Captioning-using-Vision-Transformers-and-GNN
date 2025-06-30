# Improved Medical Image Captioning with Vision Transformers, Graph Networks, CNNs, and LSTMs

This project implements and extends our research published in TenCon 2024 on generating detailed and accurate captions for medical images — particularly chest X-rays. The work explores two key pipelines:

 **Concept Detection**: Identifying key medical concepts within an image.  
 **Caption Generation**: Producing descriptive, clinically relevant text captions for medical images.

In addition to our paper’s methodology (using Vision Transformers + GCN + GPT2), this repository also provides:

- **Caption prediction using CNN networks**  
- **Caption detection using LSTM networks**  

##  Published Paper

Our full methodology and evaluation results are presented in our TenCon 2024 paper.  
🔗 [Link to TenCon 2024 paper](https://ieeexplore.ieee.org/abstract/document/10902988))


##  Repository Contents

- Jupyter notebooks (`.ipynb`) implementing:
  - Concept detection using CNN
  - Caption detection using LSTM
- Placeholder folder for future **Graph Networks implementation** (will be uploaded soon).
- Example test files demonstrating caption and concept prediction pipelines.

## Datasets

- This project uses a dataset of **~7,000 medical images**, prepared for concept detection and caption generation.
- For immediate experimentation, all code notebooks include **Kaggle links**, where you can access and fork ready-to-run notebooks with datasets preloaded.
-  ** [Data Card](https://www.kaggle.com/code/bsanjay2025/cnn-concept-detection/input) 
- For working with the **full dataset (~70,000 images)**, access it directly via Google Drive:  
  🔗 *[train](https://drive.google.com/drive/folders/1ZPqgu9YHw15DXMWQSbKl0IDAY5vMdwSz?usp=sharing),[test](https://drive.google.com/drive/folders/1pNHtS5_dreiYYKG-AfQfASbH0NU6LixX?usp=sharing)

## ⚙️ Steps to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/medical-image-captioning.git
   cd medical-image-captioning
   ```
2. **Set up Dependencies:**
   - You can run the notebooks directly in Kaggle or Colab.
   - [Kaggle Notebook Links](https://www.kaggle.com/code/bsanjay2025/caption-prediction-using-bert-embeddings),[Notebook 2](https://www.kaggle.com/code/bsanjay2025/concept-prediction-using-lstm),[Notebook 3](https://www.kaggle.com/code/bsanjay2025/cnn-concept-detection)
   - If running locally, install required Python packages from each notebook’s environment specification.

3. **Run Caption Detection (CNN/LSTM):**
   - Open the corresponding Jupyter notebook (`caption_detection_cnn.ipynb` or `caption_detection_lstm.ipynb`).
   - Replace paths to your dataset if running locally.
   - Execute all cells to train and evaluate your models.

4. **Run Concept Detection (CNN):**
   - Open `concept_detection_cnn.ipynb`.
   - Adjust dataset paths if needed.
   - Execute all cells to process images and detect concepts.

5. **Graph Networks:**
   - A notebook implementing graph-based methods using ViT + GCN + LSTM (as described in our paper) will be uploaded here.



## 📊 Evaluation

In our paper, we evaluated our models on 50 radiology image-caption pairs reviewed by 20 biomedical students using a human scoring system (1–5 scale). We also evaluated caption generation models (GPT-2) using BLEU, BERT, and ROUGE-L scores before and after fine-tuning. Key findings:

[Result]

---

## 🔎 Notes

✅ The uploaded notebooks on this repo and Kaggle include **preprocessed image datasets**, code for training, evaluation, and sample outputs.  
✅ All datasets in Kaggle notebooks are ready-to-use; simply fork and run in your own Kaggle workspace.  
✅ For the full dataset (70,000 images), download from the provided Google Drive link and adjust dataset paths in the notebooks.
