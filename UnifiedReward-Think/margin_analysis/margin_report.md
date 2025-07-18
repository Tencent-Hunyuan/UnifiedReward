## Objective

This report investigates how the **pairwise score difference** (margin) assigned by a *UnifiedReward-QWen* model influences its evaluation performance on a text-to-image generation benchmark. Specifically, we aim to:

1. Group each image pair in the GenAI-Bench test set by the absolute difference of the model’s total scores (“margin”).
2. Compute the model’s prediction accuracy within each margin group by comparing its “Image 1 vs. Image 2” preference against human votes.

## Method

The model has two kinds of inference outputs:

- **Full Chain-of-Thought (CoT):** detailed reasoning for each of the three dimensions (semantic consistency, aesthetics, authenticity) plus a final preference.  
- **Answer-Only:** a single line of text—either `Image 1 is better` or `Image 2 is better`.

Margin analysis requires the numeric scores from CoT, but the full CoT generates dozens to hundreds of tokens per sample, resulting in ~7 minutes per sample, thus making it untenable for large-scale evaluation.

To retain only the scores while eliminating the rationale text, we modified the CoT instruction (see the code for specific instructions), so that the model outputs **only** three numeric scores and their sums. Experiments show this adjustment has no noticeable effect on output quality; it merely speeds up inference while preserving accuracy.

**Full CoT example:**

```
<think>
1. Semantic consistency: 
   - Image 1 (9/10): The image shows a woman sitting on the sidewalk with rain visible, and her clothes appear wet. The hands coming out of a puddle are depicted well.
   - Image 2 (8/10): The woman is also sitting on the sidewalk in the rain, but the depiction of her clothes being wet is less clear compared to Image 1. The hands coming out of a puddle are not visible.

2. Aesthetics:
   - Image 1 (8/10): The composition is balanced with good use of colors and shading, creating an anime-style atmosphere.
   - Image 2 (7/10): The colors are vibrant, but the composition feels slightly less dynamic than Image 1.

3. Authenticity:
   - Image 1 (9/10): The details in the rain and the reflection on the wet pavement add realism to the scene.
   - Image 2 (6/10): The rain effect is present, but the overall scene lacks some realism due to the less detailed depiction of the environment.

Total score:
Image 1: 9+8+9=26
Image 2: 8+7+6=21
</think>
<answer>Image 1 is better</answer>
```

**Reformatted CoT example:**

```
<think>
SC: 9, 8
Aesthetics: 8, 7
Authenticity: 9, 6
Image 1: 9+8+9=26
Image 2: 8+7+6=21
</think>
<answer>Image 1 is better</answer>
```

## Experimental Setup

**Dataset**

- GenAI-Bench “image_generation” test split (`test_v1-00000-of-00001.parquet`)  
- Total image pairs processed: 750

**Hardware & Environment**  

- GPU: NVIDIA RTX A6000 (48 GiB VRAM)
- CUDA: 12.8
- **Working directory**: `/UnifiedReward-Think`

## Results

#### Sample Statistics

Human annotations sometimes label pairs as **both** or **tie**, indicating no preference. Since our model must choose one image per pair, we exclude these tied cases to avoid unfair penalization.

- **Total pairs:** 750  
- **Tied cases ("both"/"tie"):** 178  
- **Unparsed/anomalous cases:** 2
- **Effective pairs:** 750-178-2=570  
- **Correct predictions:** 424

**Acc. w/o ties:** 424 / 570 = **0.7439**

----

#### Margin vs. Accuracy Distribution

**Low & Medium Margin (0–7)**

|  Margin  |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |
| :------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Samples  |  63   |  188  |  107  |  62   |  35   |  37   |  12   |   7   |
| Accuracy | 0.730 | 0.745 | 0.682 | 0.677 | 0.686 | 0.811 | 0.833 | 1.000 |

**High Margin (≥ 8)**

|  Margin  |   8   |   9   |  10   |  11   |  12   |  13   |  14   | ≥ 15  |
| :------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Samples  |   6   |  12   |   8   |   6   |  11   |   4   |   6   |   6   |
| Accuracy | 0.833 | 0.833 | 0.875 | 0.833 | 0.818 | 1.000 | 1.000 | 1.000 |

- **Low-margin (0–3, ~73.7% of evaluated data):**
  Accuracy fluctuates between 0.677 and 0.745, with **no clear monotonic trend**. This indicates the model keeps uncertain in this range.

- **Mid-margin (4–7, ~16.0%):**
  Accuracy rises steadily from 0.686 (margin 4) to 1.000 (margin 7), showing a **strong correlation between increasing margin and prediction correctness**.

- **High-margin (≥ 8, ~10.3%):**

  Here, the strong positive correlation between margin and accuracy no longer holds, yet accuracy remains high (≥ 0.818, often 1.000). However, with ≤12 samples per bucket, we can’t be completely confident these findings would generalize without more data.

<img src="./logs/margin_0/stats_after_750.png" alt="stats_after_750" style="zoom:67%;" />

## Conclusions & Recommendations

1. **Automation Threshold**  
   - **Margin ≥ 5** now covers 115 of 570 effective pairs (≈ 20.2%). Within this region, per-margin accuracies range from 0.811 (margin 5) to 1.000 (margins 7, 13–16, 21–23), averaging ≈ 0.861.
   - **Action:** Auto-accept the model’s preference when margin ≥ 5; route all pairs with margin < 5 to human or secondary review.
2. **Improve Low-Confidence Performance**  
   - **Margin ≤ 3** comprises 420 of 570 effective pairs (≈ 73.7%), with accuracy fluctuating between 0.677 and 0.745, and no clear upward trend.
   - **Action:** Perform error-type analysis on these uncertain cases (e.g. semantic mismatches, artifacts) and apply targeted fine-tuning or confidence calibration (e.g. temperature scaling).