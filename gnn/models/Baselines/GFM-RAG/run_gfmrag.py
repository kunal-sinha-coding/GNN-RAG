import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gfmrag import GFMRetriever

from hydra.utils import instantiate
from gfmrag.llms import BaseLanguageModel
from gfmrag.prompt_builder import QAPromptBuilder

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig) -> None:
    import pdb; pdb.set_trace()
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    retriever = GFMRetriever.from_config(cfg)
    current_query = "Who is the president of France?"
    retrieved_docs = retriever.retrieve("Who is the president of France?", top_k=5)

    llm = instantiate(cfg.llm)
    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)

    message = qa_prompt_builder.build_input_prompt(current_query, retrieved_docs)
    answer = llm.generate_sentence(message)  # Answer: "Emmanuel Macron"
    print(answer)
    import pdb; pdb.set_trace()

main()
