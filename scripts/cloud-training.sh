echo "Submitting a training job with the AI Platform"

BUCKET_NAME=molecular-toxicity-prediction
REGION='us-east1'
MODEL_TYPE=$1
CURRENT_DATE=$(date +%Y%m%d_%H%M%S)
JOB_NAME=${MODEL_TYPE}_experiment_${CURRENT_DATE}
JOB_DIR=gs://${BUCKET_NAME}/train_experiments/${MODEL_TYPE}_experiment_${CURRENT_DATE}
MODEL_DIR=gs://${BUCKET_NAME}/model_checkpoints/${MODEL_TYPE}
SAVE_DIR=gs://${BUCKET_NAME}/callback_checkpoints/${MODEL_TYPE}
TRAIN_DATA_DIR=gs://${BUCKET_NAME}/data/train_dataset
VAL_DATA_DIR=gs://${BUCKET_NAME}/data/val_dataset
TEST_DATA_DIR=gs://${BUCKET_NAME}/data/test_dataset

gcloud ai-platform jobs submit training $JOB_NAME \
--stream-logs \
--python-version 3.7 \
--runtime-version 2.11 \
--job-dir $JOB_DIR \
--region $REGION \
--package-path ./trainer \
--module-name trainer.task \
-- \
--model_type=$MODEL_TYPE \
--train_data_dir=$TRAIN_DATA_DIR \
--val_data_dir=$VAL_DATA_DIR \
--test_data_dir=$TEST_DATA_DIR \
--model_dir=$MODEL_DIR \
--save_dir=$SAVE_DIR \
--n_epochs=10