from modules import RAG
from modules import Conversation


break_Conversation = False
llm_chat = Conversation.llm_chat()
# 초기 메세지 전달
print('start')
while (True):
    # 입력 받기
    usr_input = str(input())
    print(f'user : {usr_input}')
    # 입력 처리
    pass
    # GPT에게 입력 전달
    GPT_out, break_Conversation = llm_chat.generate_response(usr_input)
    # GPT 출력 처리

    # 출력 전달
    print(f'GPT : {GPT_out}')

    if (break_Conversation):
        print(GPT_out)
        break

# 초기 메세지와 나중의 대화 구분하기
