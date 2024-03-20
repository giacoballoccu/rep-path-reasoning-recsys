# T-REX_TextualRecEXp

If this repository IS useful for your research, we would appreciate an acknowledgment by citing our ECIR '23 paper:

```
Balloccu, G., Boratto, L., Cancedda, C., Fenu, G., Marras, M. (2023). 
Knowledge is Power, Understanding is Impact: Utility and Beyond Goals, Explanation Quality,   
and Fairness in Path Reasoning Recommendation. In: , et al. Advances in Information Retrieval. 
ECIR 2023. Lecture Notes in Computer Science, vol 13982. Springer, 
Cham. https://doi.org/10.1007/978-3-031-28241-6_1
```

# Overview
The base workflow is:
1. Dataset creation
2. Dataset **preprocessing** and **formatting** with respect to the selected model
3. Training(or hyperparameter search) of the **TransE embedding model**
4. Training(or hyperparameter search) of the **Recommender model**
5. Test of the quality of the recommender model
6. Evaluation of the metrics

## Dependencies and Installation
Required **pytorch** and **tensorflow**, both with gpu version.
Clone the repo, enter into it.
All other dependencies can then be installed by means of the env_setup.sh script located at the top level of the project
```bash
bash env_setup.sh
```
The above environment setup includes download of the datasets and their preliminary setup.

Alternatively, the data can be downloaded from the following drive:
```bash
https://drive.google.com/file/d/14WaIJHgsUuPkW8_XzN6PeBjZq15Zdt4V/view?usp=sharing
```
The setup also downloads automatically the pretrained weights for the embeddings.

# Usage
To **facilitate the use of the codebase**, a set of scripts are provided to seamlessly execute all stages of the machine learning pipeline described above.
The top level of the project contains all such scripts.

### 1. Bulk dataset creation
```bash
./build_datasets.sh
```
### 2. Bulk dataset preprocessing
```python
python3 prepare_datasets.py
```
### 3. Models
Available DATASET_NAME values **{lfm1m, ml1m}**
Available MODEL_NAME values:
- path based methods **{cafe, pgpr, ucpr}**
- knowledge aware methods **{kgat, cfkg, cke}**
- embedding methods **{transe}**

The following variables can be used. Alternatively the provided dataset and model names can be manually written as command line arguments to the scripts that follow (thus substituting the $x expression with the actual name)
```bash
export DATASET_NAME=...
export MODEL_NAME=...
```
#### 3.1 Training

```python
python3 train.py --model $MODEL_NAME --dataset $DATASET_NAME
```
#### 3.2 Hyper parameter tuning

```python
python3 gridsearch.py --model $MODEL_NAME --dataset $DATASET_NAME
```

#### 4. Test
Note, the test phase does not support testing of the embedding models, although their metrics are gathered during training.
The main purpose of the test phase is to evaluate the metrics on each embedding model.
```python
python3 test.py --model $MODEL_NAME --dataset $DATASET_NAME
```


#### 5. Evaluation
As last step, with the computed top-k or the paths predicted by the models, evaluation can be run to obtain the evaluation metrics.

Note that this step requires training of the models and at least a testing phase for path reasoning models.

This is because the path-based ones require pre-computation of paths before the topK items can be obtained. 

After paths are computed, creation of the topK is computationally inexpensive.

Knowledge aware models generate the topK by default after each epoch of training.

```python
python3 evaluate.py --model $MODEL_NAME --data $DATASET_NAME
```

Flags such as `evaluate_overall_fidelity` and `evaluate_path_quality` decide whether or not evaluate the path quality prospectives, this prospectives can be computed only for methods capable of producing reasoning paths. By default the recommendation quality metrics are evaluated. In the following section we collect the metrics currently adopted by our evaluate.py

#### 6. Metrics 
This list collects the formulas and short descriptions of the metrics currently implemented by our evaluation module. All recommendation metrics are calculated for a user top-k and the results reported on the paper are obtained as the average value across all the user base.

##### 6.1 Recommendation Quality
- **NDCG:** The extent to which the recommended products are useful for the user. Weights the position of the item in the top-k.
$$NDCG@k=\frac{DCG@k}{IDCG@k}$$ where:
$$DCG@k=\sum_{i=1}^{k}\frac{rel_i}{log_2(i+1)}=rel_1+\sum_{i=2}^{k}\frac{rel_i}{log_2(i+1)}$$
$$IDCG@k = \text{sort descending}(rel)$$
- **MMR:** The extent to which the first recommended product is useful for the user.
$$MMR = \frac{1}{\text{first hit position}}$$
- **Coverage:** Proportion of items recommended among all the item catalog.
$$\frac{| \text{Unique Recommended items}|}{| \text{Items in Catalog} |}$$
- **Diversity:** Proportion of genres covered by the recommended items among the recommended items. 
$$\frac{| \text{Unique Genres} |}{| \text{Recommended items} |}$$
- **Novelty:** Inverse of popularity of the items recommended to the user
$$\frac{\sum_{i \in I}| 1 - \text{Pop}(i) |}{| \text{Recommended items} |}$$
- **Serendipity:** Proportion of items which may be surprising for the user, calculated as the the proportion of items recommended by the benchmarked models that are not recommended by a prevedible baseline. In our case the baseline was MostPop.
$$\frac{| \text{Recommended items} \cup \text{Recommended items by most pop} |}{| \text{Recommended items} |}$$
##### 6.2 Explanation (Path) Quality
For the explanation quality metrics we suggest to refer to the original papers which provide detailed formalizations of the prospectives.
- **LIR, SEP, ETD:** [Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations](https://dl.acm.org/doi/10.1145/3477495.3532041)
- **LID, SED, PTC:** [Reinforcement Recommendation Reasoning through Knowledge Graphs for Explanation Path Quality
](https://arxiv.org/abs/2209.04954)
- **PPC** (named by us, refered in the paper as path diversity): [Fairness-Aware Explainable Recommendation over Knowledge Graphs](https://dl.acm.org/doi/10.1145/3397271.3401051)
- **Fidelity:** [Explanation Mining: Post Hoc Interpretability of Latent Factor Models for Recommendation Systems](https://dl.acm.org/doi/10.1145/3219819.3220072)
##### 6.3 Fairness 
- **Provider Fairnes:** we computed the average exposure given to products of providers in a given demographic group. The provider for MovieLens is the director of the movie while for LastFM the provider is the artist of the song.
$$\frac{\sum_{i \in I} \text{Provider Exposure}(i)}{| \text{Recommended Items} |}$$
- **Consumer Fairness:** Is defined as demographic parity for metric $Q$. $G$ is the set of demographic class (e.g. Gender group), while $g$ is the demographic groups grouping user by their attribute (e.g. Male, Female).
$$\Delta Q (G_1, G_2, Q) = \frac{1}{|G_1|} \sum_{u \in G_1} Q(u) - \mathop{} \frac{1}{|G_2|} \sum_{u \in G_2} Q(u)$$
##### 7. Hyper parameters
The hyper parameters that have been considered in the grid search are listed below, alongside a brief description and its codename used in the experiments:

###### UCPR
- `embed_size`: size of the state embedding of the employed lstm memory model, as well as the relation embedding size.
- `hidden`:  number of hidden units of each layer of the shared embedding neural network, that is used as a backbone by the actor and the critic prediction heads
###### PGPR
- `hidden`: number of hidden units of each layer of the shared embedding neural network, that is used as a backbone by the actor and the critic prediction heads
- `ent_weight`: weight of the entropy loss that quantifies entropy in the action distribution 
###### CAFE
- `embed_size`: size of the embedding of entities and relations for neural modules employed by CAFE's symbolic model
- `rank_weight`: weight of the ranking loss component in the total loss.

###### KGAT
- `adj_type`: weighting technique applied to each connection on the KG adjacency matrix A 
    -  `bilateral (bi)`, pre and post multiply A by the inverse of the square root of the diagonal matrix of out degrees of each node 
    -  `single (si)`, pre multiply A by the inverse of the of the diagonal matrix of out degrees of each node 
- `embed_size`: size of user and  entity embeddings 
- `kge_size`: size of the relation embeddings
###### CKE
- `adj_type`  (weighting technique applied to each connection on the KG adjacency matrix A )
    -  `bilateral (bi)`, pre and post multiply A by the inverse of the square root of the diagonal matrix of out degrees of each node
    -  `single (si)`, pre multiply A by the inverse of the of the diagonal matrix of out degrees of each node 
- `embed_size` (size of user and  entity embeddings)   
- `kge_size` (size of the relation embeddings)  
###### CFKG
- `lr`: learning rate
- `adj_type`: weighting technique applied to each connection on the KG adjacency matrix A 
    -  `bilateral (bi)`, pre and post multiply A by the inverse of the square root of the diagonal matrix of out degrees of each node (--adj_type bi)
    -  `single (si)`, pre multiply A by the inverse of the of the diagonal matrix of out degrees of each node (--adj_type si)
- `embed_size`: size of user and  entity embeddings
- `kge_size`: size of the relation embeddings


### Optimal hyper parameters:
Each model is configured with a set of optimal hyper parameters, according to the dataset upon which it is trained. 
In order to train a given model with customized hyper parameters, it is necessary to set them from command line while running the script train.py described in section 3.1.
Each can be set by adding as new command line arguments the pair (--param_name param_value) while also specifying the model_name and the dataset to use.
### LFM1M
###### UCPR
- embed_size  100
- hidden [64,32]
###### PGPR
- hidden [512,256]
- ent_weight 0.001
###### CAFE
- embed_size 200
- rank_weight 1.0

###### KGAT
- adj_type si
- embed_size 128
- kge_size 128
###### CKE
- adj_type  si
- embed_size 32
- kge_size 64
###### CFKG
- lr
- adj_type 0.01
- embed_size 64
- kge_size 64


### ML1M

###### UCPR
- embed_size  100
- hidden [64,32]
###### PGPR
- hidden [512,256]
- ent_weight 0.001
###### CAFE
- embed_size 200
- rank_weight 1.0

###### KGAT
- adj_type si
- embed_size 64
- kge_size 64
###### CKE
- adj_type si
- embed_size 64
- kge_size  128
###### CFKG
- lr 0.01
- adj_type  si  
- embed_size 128
- kge_size  64
