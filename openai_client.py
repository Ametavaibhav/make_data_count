from openai import OpenAI
from config import env_config, logger
import base64



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