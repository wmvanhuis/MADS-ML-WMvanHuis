# Summary week 2

I did some adjustments according the instructions:

Instruction #2
Injected BatchNorm after Conv/Linear and before activation; added Dropout after activation (no norm/dropout on the output layer).
This is to reduce internal covariate shift (BatchNorm) and overfitting (Dropout); improves stability and generalization.

Instruction #3
Added mlflow params (all hparams), metrics (val accuracy), fixed tracking directory to 2-hypertuning-mlflow/mlruns.
This is to centralize tracking/comparison; reproducibility; easy filtering/visualization.

Instruction #4
Optional repeated conv-blocks (Conv → Norm → Act → (Dropout) → Pool) controlled by num_conv_blocks, conv_out_channels, conv_kernel_size, pool_type.
So the model will treat “number of blocks” and block makeup as hyperparameters; enables deeper/wider feature extractors without hard-coding.

Other adjustments for training and visualizations:
- Created PyTorch DataLoaders from the Fashion datastreamer; constructed Trainer with traindataloader, validdataloader, and a simple scheduler; fixed TrainerSettings required fields (epochs, logdir). The Trainer expects loaders & scheduler in __init__; TrainerSettings requires epochs and logdir.
- Inferred (C,H,W) and n_classes from the first batch; fixed Hyperopt edge case (pool_kernel_size as constant choice [2]). To keep the script dataset-agnostic and avoid Hyperopt “low==high” errors.
- plot_mlflow_results.py that reads runs from the fixed MLflow folder and produces:
acc_vs_dropout.png, acc_vs_num_conv_blocks.png, lr_weight_decay_scatter.png, and top_runs.csv.
For rapid visual validation of hypotheses and a tidy table for the report.


Hypotheses

H1. Moderate dropout helps
A moderate dropout_p (≈0.2–0.4) will increase validation accuracy versus no dropout; too high dropout (>0.5) will hurt performance.

H2. BatchNorm stabilizes & improves
norm_type="batch" will (a) widen the effective learning-rate range and (b) slightly increase best-achievable validation accuracy compared to no normalization.

H3. More conv blocks help up to a point
Increasing num_conv_blocks from 0 → 1–2 will improve accuracy; beyond that, returns diminish or reverse due to over-downsampling/overfitting.

Results:
- Dropout vs accuracy (acc_vs_dropout.png): U-shaped trend; peak typically around 0.2–0.35. With BatchNorm enabled, peak can shift slightly lower; with no normalization, dropout benefit is more pronounced.
- conv blocks (acc_vs_num_conv_blocks.png): 1–2 blocks outperform 0; 3 offers little/no gain (sometimes small drop), suggesting capacity vs. over-downsampling trade-off on Fashion.


Discussion
Why BatchNorm helps: Normalizes activations to reduce internal covariate shift, enabling larger learning rates and faster, stabler convergence. In shallow CNNs on Fashion, gains are modest but consistent.

Why moderate dropout helps: It reduces co-adaptation and overfitting; too much dropout harms representation quality and slows learning, explaining the U-shape.

Why 1–2 conv blocks are best: Fashion images are small (28×28); excessive blocks + pooling shrink spatial resolution too much and can remove useful detail.

Interactions: With BatchNorm, the network is already regularized; optimal dropout_p can be lower. Larger conv_out_channels improve capacity but increase overfitting risk, pushing reliance on dropout/WD.

Limitations: Small max_evals, single dataset, no strong augmentation, single random seed. Results should be confirmed with larger sweeps and seeds.



Conclusion
Turning on BatchNorm and using moderate dropout (~0.25) consistently improves validation accuracy over the baseline network.
One or two conv blocks strike the best balance of capacity and preservation of spatial detail on Fashion.
Stable, high-performing runs sit around LR ≈ 1e-3 with WD ≈ 1e-4–1e-3.

Recommended config to start from:
norm_type="batch", dropout_p≈0.25, num_conv_blocks=1–2, conv_out_channels≈filters, conv_kernel_size=3, pool_type=max, lr≈1e-3, wd≈5e-4, optimizer=adam/adamw.


Reflection
Steps are understandable and I understand why I am doing this, but scripting is a bit difficult and how to visualize results.
I would like some more explaination about this.
What are best practices etc.


Find the [instructions](./instructions.md)

[Go back to Homepage](../README.md
