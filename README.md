# RAG Project Setup on AWS VM (Ubuntu)

This README provides a step-by-step guide to set up an AWS Ubuntu VM for the `rag_project`. It includes syncing project data, installing dependencies (MongoDB Atlas CLI, Docker), configuring Atlas deployments, and installing Python packages from `requirements.txt`.

---

## Prerequisites

- An AWS EC2 instance running Ubuntu (e.g., Ubuntu 22.04 Jammy) 20 GB root size
- SSH access to the VM (e.g., `ssh ubuntu@<VM_IP>`)
- Local `rag_project` directory at `/projects/rag_project/`

---

## Setup Instructions

### 1. Sync Project Data 

Locally run : 
```bash
rsync -avz /projects/rag_project/ ubuntu@<VM_IP>:/home/ubuntu/rag_project
```
> Pre-requisite : on VM create folder rag_project (mkdir)  
> Replace `<VM_IP>` with your AWS instance VM’s public IP.  

---

### 2. Run the script on the VM in folder rag_project
```bash
./setup.sh
```
---

### 3. Change Docker permissions 

```bash
sudo usermod -aG docker ubuntu
newgrp docker
```

---

### 4. Set Up MongoDB Atlas Local Deployment

#### Create a Local Deployment

```bash
atlas deployments setup --type local
```

> Accept the defaults when prompted.

#### Start the Deployment

```bash
atlas deployments list
atlas deployments start <deployment-name>
```

#### Connect to the Deployment

```bash
atlas deployments connect <deployment-name>
```

---

#### Activate a Virtual Environment

```bash
cd /home/ubuntu/rag_project
source venv/bin/activate
```
---

## Running the Project

```bash
python main.py
```

---

## Troubleshooting

### Disk Space Issues

```bash
df -h
sudo apt clean
sudo apt autoremove --purge
sudo rm -rf /tmp/*
pip cache purge
```

Resize EBS volume from AWS Console, then:

```bash
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
```

---

### Docker Permission Denied

```bash
sudo usermod -aG docker ubuntu
newgrp docker
```

---

### Atlas CLI Not Found

```bash
sudo apt-get install -y mongodb-atlas-cli
```

---
---

### Generating a HotpotQA provenance subset

For quick tests you can build a mini KILT corpus that only contains the documents referenced in the HotpotQA dataset:

```bash
python generate_hotpotqa_subset.py --output hotpotqa_subset.jsonl
```

Use this file as the corpus when embedding documents to speed up evaluation on a small scale.

To use the subset in the pipeline, set `CORPUS_NAME` in `config.py` to the path
of the generated JSONL file:

```python
CORPUS_NAME = "hotpotqa_subset.jsonl"
```

The data loader will detect the `.jsonl` extension and load the corpus from this
file. After updating the path run the `main.py` script to embed the subset and
evaluate retrieval performance.

### Running on PubMedQA

To evaluate the pipeline on the PubMedQA questions, change the dataset configuration in `config.py`:

```python
DATASET_NAME = "facebook/kilt_tasks"
SUBSET_NAME = "pubmedqa"
```

Then run `main.py` as usual.

### Running Each Model on a Separate VM

To avoid GPU/CPU exhaustion when using the full corpus you can evaluate one model per machine. Set the `MODELS` environment variable to the desired model before running `main.py`:

```bash
export MODELS="BAAI/bge-base-en-v1.5"
python main.py
```

Only the models listed in `MODELS` will be executed. Without this variable all predefined models run sequentially.

### Using the Full KILT Corpus

When embedding the entire Wikipedia corpus set the limits in `config.py` to `None`:

```python
CORPUS_LIMIT = None
limit = None
```

Reducing `EMBED_BATCH_SIZE` and `BATCH_SIZE` may prevent out-of-memory errors. During evaluation the script now writes each batch's results to `results_<model>.jsonl` (or the path from `OUTPUT_FILE`) so progress is saved even if a timeout occurs.
