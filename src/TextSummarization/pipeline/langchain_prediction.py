import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from src.TextSummarization.config import ConfigurationManager
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from src.TextSummarization.logger import logger


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.device = 0 if torch.cuda.is_available() else -1
        self.gen_kwargs = {
            "length_penalty": 1.0,
            "num_beams": 6,
            "max_length": 250,
            "early_stopping": True
        }

    def predict(self, text):
        logger.info(f"Received text for summarization: {text}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_dir)

        summarization_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer, device=self.device)
        llm = HuggingFacePipeline(pipeline=summarization_pipeline)

        # template = "Summarize the following dialogue, making sure to capture the main context and the most critical details or requests of the conversaton: {text}."

        template = (
            "Summarize the main points, actions, and decisions from the following conversation. "
            "Keep it brief and avoid unnecessary details.\n\n"
            "{text}"
            "\n\nSummary: "
        )

        prompt = PromptTemplate(input_variables=['text'], template=template)

        chain = prompt | llm

        print('Dialogue:', text)
        summary = chain.invoke(text)

        logger.info(f"Generated Summary: {summary}")
        print('Model Summary:', summary)

        return summary
