#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Ensure Data Directory is set if not using default "examples/finqa/data"
# export FINQA_DATA_DIR="/path/to/data"

# API Keys (Ensure these are set in your environment)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set. Training may fail if reward model needs it."
fi

# Run training
python3 train_finqa.py \
    trainer.project_name=rllm_finqa \
    trainer.experiment_name=finqa_ppo \
    trainer.total_episodes=100 \
    agent.max_steps=20 \
    rollout.temperature=0.7 \
    rllm.rollout.gen_batch_size=8 \
    rollout.n=8 \
    rollout.top_p=0.9 \
    rllm.ppo.train_batch_size=8 \
    rllm.ppo.ppo_mini_batch_size=2 \
    rllm.ppo.ppo_max_token_len_per_gpu=8192 \
    rllm.ppo.advantage_estimator=gae \
    rllm.ppo.gae_lambda=0.95 \
    rllm.ppo.kl_coeff=0.01 \
    rllm.algo.learning_rate=1e-6 \
    config_name=agent_ppo_trainer
