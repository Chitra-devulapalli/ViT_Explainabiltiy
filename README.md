# Visual Transformer Registers and Attention Maps
## Overview
This repository presents my implementation and analysis of Vision Transformers (ViTs), focusing on attention mechanisms, token norm distributions, and the impact of register tokens on model performance. The project involves visualizing attention maps, modifying ViT architectures, and evaluating their effectiveness on the ImageNet-100 dataset.

## Project Scope
### Attention Visualization
* Implemented **Chefer method** and **rollout method** for attention visualization.
* Generated attention maps for:
  * An image containing both a dog and a cat.
  * A correctly classified bird image.
  * A misclassified image from the ImageNet-100 validation set.

![submission_Q3_A](https://github.com/user-attachments/assets/7e2bd38f-65de-4097-a51e-e375dc0571db)


### Model Modification: Adding Register Tokens
* Created Model B by incorporating six register tokens into a standard ViT-B model.
* Fine-tuned the modified model on ImageNet-100, achieving over 94% accuracy.

![Submission_Model_Q3_B](https://github.com/user-attachments/assets/08bf75c5-b30a-45ca-82fc-eec95fbb0a94)

### Token Norm Analysis
* Extracted token representations at layers 1, 6, and 12.
* Computed and visualized L2 norm distributions for image patches.

### Comparative Model Analysis
* Compared attention visualizations and token norm distributions between Model A (standard ViT) and Model B (ViT with register tokens).
* Examined the effect of register tokens on feature map smoothness and classification performance.
* Model A -
    
![submission_model_A_norms](https://github.com/user-attachments/assets/d4abb948-fb84-4306-8ca1-8d4462ca3244)

* Model B -
    
![Submission_Model_B_norms](https://github.com/user-attachments/assets/ce96ce5f-04b9-4390-a043-b7cf87cdaea6)

### Theoretical Insights
* Explored interpretability in transformers.
* Investigated token relevance propagation and its impact on classification decisions.
