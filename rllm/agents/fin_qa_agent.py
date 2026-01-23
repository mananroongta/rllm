# Standard imports
import os
import sys

# Third Party Imports
from rllm.agents.tool_agent import ToolAgent
from rllm.agents.system_prompts import FIN_QA_REACT_SYSTEM_PROMPT

# Relative Imports
from rllm.tools.fin_qa_tools import Calculator, GetTableInfo, GetTableNames, SQLQuery


class FinQAAgent(ToolAgent):
    """
    An agent that can answer questions about financial statements.
    """

    def __init__(
        self,
        system_prompt=FIN_QA_REACT_SYSTEM_PROMPT,
        parser_name="qwen",
    ):
        """
            Initialize the FinQAAgent.

            Args:
                system_prompt: System prompt for the agent.
                    Default: FIN_QA_REACT_SYSTEM_PROMPT 
                parser_name: Name of the parser to use for tool calls.
                    Default: "qwen", same default as ToolAgent.
        """
        
        # Initialize the tool map.
        fin_qa_tool_map = {
            "calculator": Calculator,
            "get_table_info": GetTableInfo,
            "get_table_names": GetTableNames,
            "sql_query": SQLQuery
        }

        # Initialize the ToolAgent.
        super().__init__(
            system_prompt=system_prompt,
            parser_name=parser_name,
            tool_map=fin_qa_tool_map,
        )
