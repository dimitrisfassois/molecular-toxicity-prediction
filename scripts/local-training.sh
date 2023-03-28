echo "Submitting a local training job"

MODEL_TYPE=$1

BUCKET_NAME=molecular-toxicity-prediction
CURRENT_DATE=$(date +%Y%m%d_%H%M%S)
JOB_DIR=gs://${BUCKET_NAME}/train_experiments/${MODEL_TYPE}_experiment_${CURRENT_DATE}
MODEL_DIR=/tmp/model_checkpoints/${MODEL_TYPE}
SAVE_DIR=/tmp/callback_checkpoints/${MODEL_TYPE}
TRAIN_DATA_DIR=gs://${BUCKET_NAME}/data/train_dataset
VAL_DATA_DIR=gs://${BUCKET_NAME}/data/val_dataset
TEST_DATA_DIR=gs://${BUCKET_NAME}/data/test_dataset

gcloud ai-platform local train \
--package-path trainer \
--module-name trainer.task \
-- \
--job-dir $JOB_DIR \
--model_type=$MODEL_TYPE \
--train_data_dir=$TRAIN_DATA_DIR \
--val_data_dir=$VAL_DATA_DIR \
--test_data_dir=$TEST_DATA_DIR \
--model_dir=$MODEL_DIR \
--save_dir=$SAVE_DIR \
--n_epochs=30