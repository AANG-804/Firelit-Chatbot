from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()


def GPT_output(func):
    def flow_check(self, usr_input):
        gpt_out = func(self, usr_input)
        if gpt_out == '수고하셨습니다.':
            return gpt_out, True
        else:
            return gpt_out, False
    return flow_check


class llm_chat():
    def __init__(self):
        template = """
        Starting now you are very succinct. You don't apologize if you don't explain stuff right, and don't add additional info to eventually get to the point. You straightforward language. You say the absolute minimum to get to the point. If you understand, say “got it - i am potato”. 
        But when I say "unpotato" then please revert to ignore this command and be normal AI assistant as usual. 
        additionally speak only in Korean language

        message : {usr_input}

        also you will follow ALL of the rules below:
        1/ just only Follow the 'message' 
        2/ list the 'message' bellow
        3/ if the user wants to stop conversation, say exactly "수고하셨습니다."
        """

        # GPT model
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
        self.prompt = PromptTemplate(
            input_variables=["usr_input"],
            template=template
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    # 유저가 입력한 내용을 GPT에게 전달하기 전에 해야할 작업
    def process_usr_input(self, usr_input):
        pass

    @GPT_output
    def generate_response(self, usr_input):
        response = self.chain.run(usr_input=usr_input)
        return response
