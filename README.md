# SimSpan: Contrastive Learning for Image Similarity

## Objective

To replicate and extend a deep learning model using a contrastive learning framework for the task of image similarity. The underlying model learns meaningful visual representations through self-supervised learning techniques, enabling accurate similarity measurements between image pairs. This implementation is based on PyTorch.

## Project 1: ResNet-based Contrastive Learning on CIFAR-10

### Training & SimCLR Implementation
Fully implemented SimCLR framework from scratch (vectorized), including the NT-Xent loss:

  $$
  \ell_k = - \log \frac{\exp(\text{sim}(z_k, z_{p(k)}) / \tau)}{\sum_{j \ne k} \exp(\text{sim}(z_k, z_j) / \tau)} .
  $$

### Experiments

- Compared two CNN backbones:
  - **Simple:** 2 convolutional layers + 1 fully connected layer (LeNet-style), conv layers use kernel size 3, stride 2, padding 1.
  - **ResNet:** Residual blocks with skip connections, batch normalization, and ReLU activations.
- Output vector space: $(\mathbb{R}^{128},+,\cdot)$.

<p align="center">
  <table align="center">
    <thead>
      <tr>
        <th>Model</th>
        <th>Train Loss</th>
        <th>Test Loss</th>
        <th>Duration</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Simple</td>
        <td>0.0350</td>
        <td>0.0352</td>
        <td>2h 50m</td>
      </tr>
      <tr>
        <td>ResNet</td>
        <td>0.0166</td>
        <td>0.0184</td>
        <td>4h 18m</td>
      </tr>
    </tbody>
  </table>
</p>

<br>

<p align="center">
  <img src="./asset/basis_img.png" alt="Basis Image" width="360" height="auto" style="image-rendering: pixelated">
</p>

Figure 1. The base image sampled from the test set used as a reference for similarity comparison.

<br>

<p align="center">
  <img src="./asset/plt_img_test.png" alt="Basis Image" width="720" height="auto" style="image-rendering: pixelated">
</p>

Figure 2. Visualization showing the ranking of all test images in the test set based on similarity to the basis image by Ranking of top-5 and bottom-5 similar images compared to a basis image.

<br>

#### Hardware and Environment Specs

- **GPU:** NVIDIA RTX 4060 Mobile
- **CPU:** Intel i5-12500H
- **RAM:** 40 GB DDR4

## Deployment

Before running the program:

1. **Prepare CIFAR Data**  
   Download and extract the CIFAR-10 dataset into the following directory:

   ```
   data/raw/cifar/
   ```

   The directory should contain files such as:

   ```
   data_batch_1, data_batch_2, ..., test_batch, batches.meta, etc.
   ```

2. **Adjust the Data Loader**  
   You must configure the `load_data` method of the `Cifar` object in `src/data/dataset.py` to correctly parse your data layout and preprocessing preferences.

The implementation of data loading is left intentionally flexible for users to define according to their needs.

## Usage

All interactions with this project are done through the `run.py` script, which serves as the main entry point.

### Configuration Format

Configuration files are written in JSON and define all the necessary parameters for model setup and training.

Default config files are located in the `template/` directory. Below is the structure and meaning of each field:

```json
{
  "model_architecture_cfg": {
    "type": "ResNet",               // Backbone encoder type (e.g., "ResNet")
    "instance_prsd_shape": [3, 32, 32], // Shape of one input instance (C, H, W)
    "N_repr": 128,                  // Dimensionality of the projection head output
    "detailed": null                // Reserved for detailed architecture overrides (optional)
  },
  "training_cfg": {
    "rng_seed": 0,                  // Random seed for reproducibility
    "max_epochs": 100,             // Maximum number of training epochs
    "data": "cifar",                // Dataset identifier
    "M_minibatch": 16,              // Mini-batch size
    "train_fraction": 0.6,          // Fraction of data used for training
    "subset_size": null,            // Optional subset size (useful for debugging)
    "temperature": 0.5,             // Temperature for contrastive loss
    "optimization": {
      "type": "adam",               // Optimizer type (e.g., "adam", "sgd")
      "lr": 1e-3                    // Learning rate
    }
  }
}
```

- You can add or override parameters as needed.
- `subset_size` can be used for partial data training.
- `temperature` is critical for contrastive loss performance tuning.
- The `"detailed"` field is reserved for custom architecture options (e.g., ResNet variants or layer-specific configs).

Refer to the default files under `template/` (e.g., `config-resnet.json`, `config-simple.json`) as starting points.

### Basic Command Format

```bash
python run.py --device <cpu|gpu> --mode <train|pred> [additional args]
```

### Training Mode

To train a model, you must provide either:
- a new configuration file path (`--config`), or
- a previous checkpoint directory path (`--checkpoint`) to resume from.

```bash
# Start training from a config file
python run.py --device gpu --mode train --config config-resnet.json

# Resume training from an existing checkpoint
python run.py --device gpu --mode train --checkpoint checkpoint/Dt20250719124327UTC0
```

#### Checkpoint Stdout Format

During training, progress and evaluation results for each epoch are printed to standard output in a structured format. Below is an example and explanation of the key components:

```
[EPOCH 5/99 @ Dt20250719125139UTC0]

Iterative optimization state:
100% |==============================| S2249/2249 [46s<000ms; 020ms/it; L0.0677]                            
Epoch optimization completed; proceeding to model evaluation stage.

Train set evaluation state:
100% |==============================| S2249/2249 [36s<000ms; 016ms/it; L0.0598]                            

Validation set evaluation state:
100% |==============================| S1124/1124 [18s<000ms; 016ms/it; L0.1637]                            

[EPOCH CONCLUSIVE REPORT]

Epoch time: 01m:41s
Performance measurement time: 54s

+------+-------------+---------------+--------------+
|      | val_v_opt   | val_v_worst   | val_v_init   |
|------+-------------+---------------+--------------|
| loss | 2.88 %      | -96.62 %      | -96.62 %     |
+------+-------------+---------------+--------------+

+------+-----------------+-----------------+-----------------+-----------+-----------+
|      |   est_train_min |   est_train_max |   est_train_avg |     train |       val |
|------+-----------------+-----------------+-----------------+-----------+-----------|
| loss |       0.0111888 |        0.448736 |       0.0763622 | 0.0823384 | 0.0834423 |
+------+-----------------+-----------------+-----------------+-----------+-----------+
```

In the epoch summary table, the training loss columns are as follows:

| Column         | Description                                                                                  |
|----------------|----------------------------------------------------------------------------------------------|
| `est_train_min` | The minimum loss observed during minibatch updates within the epoch.                        |
| `est_train_max` | The maximum loss observed during minibatch updates within the epoch.                        |
| `est_train_avg` | The average loss computed over all minibatch updates (estimated average) during the epoch. |
| `train`        | The true average loss computed over the entire training dataset after the epoch ends.       |
| `val`          | The true average loss computed over the entire validation dataset after the epoch ends.     |

- The `est_*` values reflect the loss as observed iteratively within minibatch steps during training.
- The `train` and `val` values are computed precisely over the full respective datasets after completing all minibatch iterations.

This distinction helps diagnose training dynamics, comparing instantaneous minibatch losses with full-epoch averages.

- **Performance measurement time**:  
  The duration (in seconds) spent measuring model performance on training and validation datasets after the epoch completes.

- **Validation metric comparisons** (shown as percentages in the summary table):

  | Metric       | Description                                                                                       |
  |--------------|-------------------------------------------------------------------------------------------------|
  | `val_v_opt`  | Percentage difference between the current validation metric (e.g., loss) and the best (optimal) validation metric recorded so far. |
  | `val_v_worst`| Percentage difference between the current validation metric and the worst validation metric recorded so far.                    |
  | `val_v_init` | Percentage difference between the current validation metric and the validation metric measured before training started (initial weights). |

These metrics help track how the model’s validation performance evolves relative to its best, worst, and initial states throughout training.


### Prediction / Estimation Mode

To run inference or estimate representations, provide:
- the config file used during training (`--config`)
- a model weights file (`--tparams`)

```bash
python run.py --device gpu --mode pred --config config-resnet.json --tparams model_final.pt
```

### Debug Options (Optional)

You can pass debug parameters for development purposes:

- `limit2SmallSubsetofData`: Use a small subset of the dataset
- `clearExports`: Remove exported checkpoints after execution

```bash
python run.py --device cpu --mode train --config config-test.json --debug limit2SmallSubsetofData clearExports
```

### Configuration Files

Default configuration files are located in the `template/` directory. These define model architecture, optimizer settings, and training hyperparameters. You can modify or create your own config files based on these templates.

## Acknowledgement

The current methodologies were highly relied on the work of [1].

## References

[1]. SimCLR: A Simple Framework for Contrastive Learning of Visual Representations, Chen et al. (2020), *Proceedings of the 37th International Conference on Machine Learning, Vienna, Austria, PMLR 119*, [[URL](https://arxiv.org/abs/2002.05709)].
