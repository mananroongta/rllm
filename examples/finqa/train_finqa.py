import hydra
import os
import sys

# Allow running from anywhere in repo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.workflow import TerminationEvent, TerminationReason

# Centralized imports from rllm
from rllm.agents.fin_qa_agent import FinQAAgent
from rllm.environments.fin_qa_environment import FinQAEnvironment


class FinQAWorkflow(MultiTurnWorkflow):
    """MultiTurnWorkflow with reward logging"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)

        self.agent.update_from_env(observation, 0, False, info)

        # Calculate max context once (max_prompt_length + max_response_length)
        max_model_len = self.rollout_engine.max_prompt_length + self.rollout_engine.max_response_length
        min_response_buffer = 1000  # Minimum tokens to reserve for model response

        for _ in range(1, self.max_steps + 1):
            # Check if conversation is approaching context limit
            if hasattr(self.rollout_engine, 'chat_parser'):
                # verl backend - use chat_parser
                prompt = self.rollout_engine.chat_parser.parse(
                    self.agent.chat_completions,
                    add_generation_prompt=True,
                    is_first_msg=True
                )
                prompt_length = len(self.rollout_engine.tokenizer.encode(prompt, add_special_tokens=False))
            else:
                # Tinker backend - use tokenizer directly
                try:
                    prompt_ids = self.rollout_engine.tokenizer.apply_chat_template(
                        self.agent.chat_completions,
                        add_generation_prompt=True,
                        tokenize=True
                    )
                    prompt_length = len(prompt_ids)
                except Exception:
                    # Fallback if tokenizer fails or template missing
                    prompt_length = 0

            if prompt_length > max_model_len - min_response_buffer:
                raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

            output: ModelOutput = await self.rollout_engine.get_model_response(self.agent.chat_completions, application_id=uid, enforce_max_prompt_length=False, **kwargs)
            response = output.text

            action = self.agent.update_from_model(response)

            # Store model_output on step for Tinker training
            if self.agent.trajectory.steps:
                self.agent.trajectory.steps[-1].model_output = output

            next_obs, reward, done, info = await self.run_in_executor(self.env.step, action.action)
            
            self.agent.update_from_env(next_obs, reward, done, info)

            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)
    
    def assign_episode_correctness(self, episode: Episode) -> None:
        """Use is_correct from Reward, before relying on default option"""
        # Check if last step has is_correct from environment
        if episode.trajectories and episode.trajectories[0].steps:
            is_correct = episode.trajectories[0].steps[-1].info.get("is_correct")
            if is_correct is not None:
                episode.is_correct = is_correct
                return
        
        super().assign_episode_correctness(episode)
    
    def collect_metrics(self, episode: Episode) -> None:
        super().collect_metrics(episode)
        # Added metadata from last step -> for wandb logging
        if episode.trajectories and episode.trajectories[0].steps:
            metadata = episode.trajectories[0].steps[-1].info.get("metadata", {})
            if metadata:
                episode.metrics.update(metadata)


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Ensure dataset is loaded
    train_dataset = DatasetRegistry.load_dataset("finqa", "train")
    val_dataset = DatasetRegistry.load_dataset("finqa", "test")

    if not train_dataset:
        print("FinQA dataset not found. Please run prepare_finqa_dataset.py first.")
        return

    config.rllm.workflow.use_workflow = True
    
    # Optional: override rollout settings via config or here
    # config.rllm.rollout.temperature = 0.0

    trainer = AgentTrainer(
        workflow_class=FinQAWorkflow,
        workflow_args={
            "agent_cls": FinQAAgent,
            "env_cls": FinQAEnvironment,
            "max_steps": 20,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
