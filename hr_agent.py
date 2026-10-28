
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables from .env
load_dotenv()


async def main():
    # Configure the OpenAI model
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    try:
        # Create the HR assistant agent
        agent = AssistantAgent(
            name="hr_assistant",
            model_client=model_client,
            system_message="You are an HR assistant."
        )

        # Run the agent and display its streaming response
        await Console(
            agent.run_stream(
                task=(
                    "Write a 5-line job description for a "
                    "Junior Python Developer."
                )
            )
        )

    finally:
        # Close the model client
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
