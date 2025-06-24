from openai import OpenAI
from config import env_config, logger
import base64
import json
import re



class OpenAIClient:
    def __init__(self):
        self._client = OpenAI(api_key=env_config['OPENAI_KEY'])

    @staticmethod
    def image_to_b64(image_path):
        with open(image_path, 'rb') as f:
            return base64.b64decode(f.read()).decode("utf-8")


    def _prepare_input(self, query, image_path):
        """
        Only considers local images/paths
        # Not adding system prompt for now
        """
        if image_path:
            input_arr = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": query
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/{image_path.split('.')[-1]};bas64,{self.image_to_b64(image_path)}"
                        }
                    ]
                }
            ]
        else:
            input_arr = [
                {
                    "role": "user",
                    "content": query
                }
            ]
        return input_arr


    def ask_openAI(self, query, image_path=None, model_name='gpt-4.1', stream=False):
        model_input = self._prepare_input(query, image_path)

        response = self._client.responses.create(
            model=model_name,
            input=model_input,
            stream=stream
        )
        return response

    @staticmethod
    def extract_response(response):
        return response.output_text


    @staticmethod
    def extract_codeBlockData(text, returnInput=False):
        """
        given text, extract data that may be present inside a code block (enclosed withing ``` ```)
        only returns the data within the very first code block encountered.

        Args:
            text (str): string from which the data needs to be extracted
            returnInput (bool): if True, will return ib if there was no code block found.

        Returns:
            extracted_data (str/None): string if any data was extracted. None, in case of failure/No data found.
        """
        try:
            ## also dealing with incomplete code blocks (code block created, but not ended)
            if text.count("```") == 1:
                text += "```"
            pattern = r"```(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            text = matches[0].strip()
        except Exception as e:
            logger.error(f"Error @extract_codeBlock(): {str(e)}")
            if not returnInput:
                text = None
        return text

    @staticmethod
    def extract_jsonCodeBlock(text, strict_flag=False):
        """
        given a json string (text), extract it into a dict;
        performs a strip('json'), and strip() on the string before a json.loads

        Args:
            text (str): text cotaining the json string
            strict_flag (bool): flag for 'strict' kwarg for json.loads; will be okay with control characters in the string

        Returns:
            text (dict/None): json object as a dict, if the extraction was successful; None, otherwise.
        """
        try:
            ### removing json at the beginning/end of the text
            text = text.strip("```").strip('json').strip('JSON').strip()
            text = json.loads(text, strict=strict_flag)
        except Exception as e:
            logger.error(f"Error @extract_jsonCodeBlock(): {str(e)}")
            text = None
        return text
