# CODS
the code implementation of Capability-Oriented Data Selection ( CODS )

## Layout

```
CODS
├── data
│   ├── meta_data.json
│   ├── label.py
│   └── model.py
├── evaluate
│   ├── extract_embedding.py
│   ├── get_few_shot.py
│   ├── load_data.py
│   ├── save_filter_data.py
│   ├── test.py
│   └── train_lora.py
├── models
├── README.md
└── src
    ├── dataloader.py
    ├── main.py
    └── model.py
```

## Train

### data prepareation

Firsty, you need to prepare the annotated reference datasets. 

1. You can directly use our Annotated reference datasets [meta_data.jsonl](data/meta_data.json) mentiond in our paper 
2. Or you can collect your own datasets and label them via external models. We provide some programs which can be directly used to do labeling work. You can refer to [label.py](data/label.py) and [model.py](data/model.py)

Secondly, you need to run some MoE method and MLLM data-selection method on your annotated reference datasets to collect their routing factors, hidden states and gradients. You can store these results in `data/` which will be used during training.

We use [TIVE](https://github.com/simplelifetime/TIVE/tree/master), [COINCIDE](https://github.com/G-JWLee/COINCIDE_code) and [MOVA](https://github.com/TempleX98/MoVA) as the chosen method. You can refer to their repo for more detailed information.

### dataloader and configuration

Based on the method you choose in the previous step, you may need to modify the [dataloader.py](src/dataloader.py) to fit in the gradients, hiddent states and routing factors format. If you use the same method as ours, you don't need to change anything.

We implement two-stage training paradigm, decouples the
ability space learning into two stages: Representation Learn-
ing and Space Adjustment. You can change the detailed condifguration in [main.py](src/main.py)

~~~python
# Two-stage training configuration
config = {
    # Model parameters
    'projection_dim': 256,
    'hidden_dim': 256,
    'num_blocks': 4,
    'sphere_dim': 64,  # Lower sphere dimension for improved stability
    'dropout_rate': 0.2,
    
    # Training parameters
    'stage1_lr': 2e-3,      # Stage 1 learning rate
    'stage2_lr': 5e-4,      # Stage 2 learning rate
    'stage1_epochs': 20,    # Stage 1 epochs
    'stage2_epochs': 6,     # Stage 2 epochs
    'batch_size': 32,
    
    # Loss weights
    'contrastive_weight': 0.1,   # Contrastive loss weight
    'repulsion_weight': 0.05,    # Repulsion loss weight
}

# input files path configuration
gradients_file = "your_gradients_file.pt"
routing_weights_file = "your_routing_weights_file.pt"
meta_file = "your_meta_file.json"
~~~

### train the projector

After dealing with all the preparatory work, you can simply running the following command to start training 

```bash
cd src
python main.py 
```

## Evaluation

After you have prepared training data, you just need to run the following command to train the LoRA model:

First, you should prepare the training environment. We use ms-swift as our training framework, so you need to install it first. You can find the installation instructions in the [ms-swift repository](https://github.com/modelscope/ms-swift). Note that you should install the full capability version of ms-swift, which includes the training capability.

```bash
python train_lora.py --method CODS
```

This command will start the training process using the CODS method. There are ["random", "repeat", "cods", "tive", "coincide"] methods available, and you can choose one of them at a time.

After the training is complete, you can evaluate the model using the following command:

```bash
python test.py --method CODS
```

You can also change the checkpoint path used in test.py to get the evaluation results of different checkpoints. The default checkpoint path is set to `checkpoint-90`.

All these training and evaluation are validated on a single NVIDIA RTX 4090 GPU.