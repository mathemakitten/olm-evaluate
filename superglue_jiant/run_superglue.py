import sys

import os

import jiant.utils.python.io as py_io
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader

# See https://github.com/nyu-mll/jiant/blob/master/guides/tasks/supported_tasks.md for supported tasks
# TASK_NAME = "mrpc"
TASKS = "boolq,mrpc,copa,multirc,record,rte,wic,wsc"

# See https://huggingface.co/models for supported models
# HF_PRETRAINED_MODEL_NAME = "roberta-base"
HF_PRETRAINED_MODEL_NAME = "Tristan/olm-bert-base-uncased-oct-2022"

# Remove forward slashes so RUN_NAME can be used as path
MODEL_NAME = HF_PRETRAINED_MODEL_NAME.split("/")[-1]
RUN_NAME = f"simple_superglue_{MODEL_NAME}"
EXP_DIR = "./superglue_jiant/exp"
DATA_DIR = "./superglue_jiant/exp/tasks"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

for task in TASKS.split(","):
    downloader.download_data([task], DATA_DIR)

args = simple_run.RunConfiguration(
    run_name=RUN_NAME,
    exp_dir=EXP_DIR,
    data_dir=DATA_DIR,
    hf_pretrained_model_name_or_path=HF_PRETRAINED_MODEL_NAME,
    tasks=TASKS,
    train_batch_size=1024,
    num_train_epochs=1,
    do_save_last=True,

)
simple_run.run_simple(args)

args = simple_run.RunConfiguration.from_json_path(os.path.join(EXP_DIR, "runs", RUN_NAME, "simple_run_config.json"))
simple_run.run_simple(args)