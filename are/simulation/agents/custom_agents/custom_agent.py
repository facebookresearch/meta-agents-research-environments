from are.simulation.agents.default_agent.base_agent import BaseAgent, ConditionalStep
from are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)


class MyCustomAgent(BaseAgent):
    def __init__(self, llm_engine, **kwargs):
        # Custom system prompts
        system_prompts = {
            "system_prompt": """You are a helpful AI assistant specialized in email management.
            Your goal is to help users organize and respond to emails efficiently.

            Available tools: <<tool_descriptions>>

            Always think step by step and explain your reasoning."""
        }

        # Custom action executor
        action_executor = JsonActionExecutor(llm_engine=llm_engine)

        # Custom conditional steps
        pre_steps = [
            ConditionalStep(
                condition=lambda agent: agent.iterations == 0,
                function=self.initial_setup,
                name="initial_setup",
            )
        ]

        super().__init__(
            llm_engine=llm_engine,
            system_prompts=system_prompts,
            action_executor=action_executor,
            conditional_pre_steps=pre_steps,
            max_iterations=15,
            **kwargs,
        )

        self.name = "email_specialist_agent"

    def initial_setup(self):
        """Custom initialization logic."""
        self.logger.info("Initializing email specialist agent...")
        # Add custom setup logic here
