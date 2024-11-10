import os
from openai import OpenAI
from groq import Groq
import openai

MAX_LEN = 3000


def execute_api_call(client, model, messages, temperature, max_tokens):
    response = None
    while response is None:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        except Exception as e:
            print(f"API call failed with error: {e}. Retrying...")
    return response


class PromptExecutor:
    def __init__(self, prompt_format, model, temperature=0, max_tokens=200):
        self.prompt_format = prompt_format
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def execute_prompt(self, prompt):
        client = None
        if "gpt" in self.model:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model in ["mixtral-8x7b-32768", "gemma2-9b-it", "llama3-70b-8192", "llama3-8b-8192"]:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        else:
            openai.api_key = "EMPTY"
            openai.base_url = "http://localhost:8000/v1/"
            if prompt[0]["role"] == "system":
                prompt[1]["content"] = prompt[0]["content"] + prompt[1]["content"]
                prompt = prompt[1::]
            response = openai.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        return execute_api_call(client, self.model, prompt, self.temperature,
                                self.max_tokens) if client is not None else response


class PromptFormat:

    def __init__(self, topk=5):
        self.topk = topk

    def format_prompt(self, inp):
        raise NotImplementedError

    def format_history(self, history):
        return "\n".join([f"{'Student:' if i % 2 else 'Customer:'} {s}" for i, s in enumerate(history)]) + "\n"

    def format_experience(self, prev_exp):
        if not prev_exp:
            return "No previous experience\n"
        return "\n".join([f"Experience {j + 1}:\n" + "\n".join(
            [f"{'Student:' if i % 2 else 'Customer:'} {s}" for i, s in enumerate(exp)]) for j, exp in
                          enumerate(prev_exp)]) + "\n"

    def format_choices(self, choices):
        return "\n".join([f"{i + 1}. {c}" for i, c in enumerate(choices[:self.topk])]) + "\n"

    def parse_response(self, response):
        return response.choices[0].message.content


class GPTChooses_or_Vetos(PromptFormat):
    def __init__(self, topk=5):
        super().__init__(topk)
        self.topk = topk

    def format_prompt(self, history, subject, problem, choices, posttest=False, prev_exp=None):
        formated_history = self.format_history(history)
        formated_prev_exp = self.format_experience(prev_exp)
        formated_choices = self.format_choices(choices)
        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}Student top choices:\n{formated_choices} {question} if there is a suitable action between the student choice answer with the choice number. you can first reason about your choice in less than 50 words. Do not forget to put ### after your reasoning finishes.Then write your chosen action.Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then. If you do not find any suitable actions between the student's top choices write 0. \nexample output1:\nreason: we need to explore the baby's symptoms more.\n###\n1\nexample output2:\nreason: I cannot find a suitable answer between the student's picks.\n###\n0"
            }
        ]


class GPTChooses_or_Recs(PromptFormat):
    def __init__(self, topk=5):
        super().__init__(topk)
        self.topk = topk

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, choices,
                      posttest=False, prev_exp=None):
        formated_history = self.format_history(history)
        formated_prev_exp = self.format_experience(prev_exp)
        formated_choices = self.format_choices(choices)
        formated_valid_subjects = self._format_list(valid_subjects)
        formated_valid_topics = self._format_list(valid_topics)
        formated_valid_causes = self._format_list(valid_causes)
        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        available_actions = self._available_actions(posttest)
        formated_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        output = "1. choose(cause1)\n2. choose(cause2)" if posttest else "1. ask(x,y)\n2. answer()"

        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}Student top choices:\n{formated_choices} available actions:{available_actions}\n{formated_options} {question} if there is a suitable action between the student choice answer with the choice number. you can first reason about your choice in less than 50 words. Don't forget to put ### after your reasoning finishes.Then write your chosen action.Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then. If you do not find any suitable actions between the student's top choices recommend {5 if not posttest else 2} actions from the list to the student. Write whether you are doing choose or recommend followed by $$$.\nexample output1:\nchoose\n$$$\nreason: we need to explore the baby's symptoms more.\n###\n1\nexample output2:\nrecommend\n$$$\nreason: we need to explore the baby's symptoms more but the student have not included it in their choice.\n###\n{output}"
            }
        ]


class CLINChooses_or_Vetos(PromptFormat):
    def __init__(self, topk=5):
        super().__init__(topk)
        self.topk = topk

    def format_prompt(self, history, subject, problem, choices, summary, posttest=False, prev_exp=None,
                      ):
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                         f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for completing your task:\n"
        user_em = "if there is a suitable action between the student choices answer with the choice number. If you do not find any suitable actions between the student's top choices write 0.\n First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for your decision. If no Learning is selected, construct the rationale based on the observation history.\nFormat your response as follows:\nWrite the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale of at most 50 words. Finally, write ### followed by number.\n"

        formated_history = self.format_history(history)
        formated_prev_exp = self.format_experience(prev_exp)
        formated_choices = self.format_choices(choices)

        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}{summary_prompt}{summary}Interaction History:\n{formated_history}Student top choices:\n{formated_choices} {question} {user_em}\nexample output1:\n1, 3\n$$$\nreason: we need to explore the baby's symptoms more.\n###\n1\nexample output2:\n1, 3\n$$$\nreason: I cannot find a suitable answer between the student's picks.\n###\n0"
            }
        ]


class CLINChooses_or_Recs(PromptFormat):
    def __init__(self, topk=5):
        super().__init__(topk)
        self.topk = topk

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, choices, summary,
                      posttest=False, prev_exp=None):
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                         f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for completing your task:\n"
        output = "1. choose(cause1)\n2. choose(cause2)" if posttest else "1. ask(x,y)\n2. answer()"
        formated_history = self.format_history(history)
        formated_prev_exp = self.format_experience(prev_exp)
        formated_choices = self.format_choices(choices)
        formated_valid_subjects = self._format_list(valid_subjects)
        formated_valid_topics = self._format_list(valid_topics)
        formated_valid_causes = self._format_list(valid_causes)
        available_actions = "1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        formated_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        user_em = f"if there is a suitable action between the student choices answer with the choice number. If you do not find any suitable actions between the student's top choices recommend {5 if not posttest else 2} actions from the list to the student.\n First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for your decision. If no Learning is selected, construct the rationale based on the observation history.\nFormat your response as follows:\nWrite the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write *** followed by the rationale of at most 50 words. Next, write $$$ followed by whether you are \"choose\" or \"recommend\". Finally, write ### followed by your recommended actions or chosen action.\n"
        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure.Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}{summary_prompt}{summary}Interaction History:\n{formated_history}Student top choices:\n{formated_choices} available actions:{available_actions}\n{formated_options} {question} {user_em}example output1:\n1, 3\n***\nchoose\n$$$\nreason: we need to explore the baby's symptoms more.\n###\n1\nexample output2:\n1, 3\n***\nrecommend\n$$$\nreason: we need to explore the baby's symptoms more but the student have not included it in their choice.\n###\n{output}"}
        ]


class CLINRecs_with_fallback(PromptFormat):
    def __init__(self, num_of_recs=5):
        super().__init__(num_of_recs)
        self.num_of_recs = num_of_recs

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, previous_suggestions,
                      summary, posttest=False,
                      prev_exp=None):
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                         f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for completing your task:\n"
        user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                                Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale of at most 50 words. Finally, write ### followed by your list of recommended actions.
                                """
        formated_history = self.format_history(history)
        formated_prev_exp = self.format_experience(prev_exp)
        formated_valid_subjects = self._format_list(valid_subjects)
        formated_valid_topics = self._format_list(valid_topics)
        formated_valid_causes = self._format_list(valid_causes)
        available_actions = "1.choose(cause)[The most probable reason is cause]" if posttest else " 1. ask(subject,topic)[I want to ask about subject’s topic.] 2. answer()[I want to suggest the cause behind the customer’s problem.]\n"
        formated_options = f"valid causes: {formated_valid_causes}" if posttest else f"valid subjects: {formated_valid_subjects}\nvalid topics: {formated_valid_topics}\n"
        output = "1. choose(cause1)\n2. choose(cause2)" if posttest else "1. ask(x,y)\n2. answer()"
        prompt = [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will recommend actions for the student to do. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then. Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}{summary_prompt}{summary}Interaction History:\n{formated_history}available actions:{available_actions}\n{formated_options}What are your top {2 if posttest else self.num_of_recs} recommended actions from the action list? you can first reason about your recommendation in less than 50 words. Don't forget to put ### after your reasoning finishes. Then write your suggested actions. {user_em}\nexample output:\n1, 3\n$$$\nreason: r\n###\n1"
            }
        ]
        user = f"The student would prefer not to follow your suggestions. Recommend new actions to the student based on the valid actions list. \noutput format:\nreason: r\n###\n{output}"
        if len(previous_suggestions) > 0:
            appendix = []

            for i, s in enumerate(previous_suggestions):
                appendix.append({
                    "role": "assistant",
                    "content": f"{s}"
                })
                appendix.append({
                    "role": "user",
                    "content": user
                })
            prompt += appendix

        return prompt


class GPTChooses(PromptFormat):
    def __init__(self, topk=5):
        super().__init__(topk)
        self.topk = topk

    def format_prompt(self, history, subject, problem, choices, posttest=False, prev_exp=None):
        formated_history = self._format_history(history)
        formated_prev_exp = self._format_prev_exp(prev_exp)
        formated_choices = self._format_choices(choices, self.topk)

        question = "What should the student answer?" if posttest else "What should the student choose to do?"

        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}Student top choices:\n{formated_choices} {question} answer with the choice number you can first reason about your choice in less than 50 words. Don't forget to put ### after your reasoning finishes.Then write your chosen action.Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then.\nexample output:\nreason: we need to explore the baby's symptoms more.\n###\n1"
            }
        ]


class GPTRecs(PromptFormat):
    def __init__(self, num_of_recs=5):
        super().__init__(num_of_recs)
        self.num_of_recs = num_of_recs

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, posttest=False,
                      prev_exp=None):
        formated_history = self._format_history(history)
        formated_prev_exp = self._format_prev_exp(prev_exp)
        available_actions = self._available_actions(posttest)
        formated_options = self._format_options(valid_subjects, valid_topics, valid_causes, posttest)

        output = "1. choose(cause1)\n2. choose(cause2)" if posttest else "1. ask(x,y)\n2. answer()"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will recommend actions for the student to do. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}available actions:{available_actions}\n{formated_options}What are your top {2 if posttest else self.num_of_recs} recommended actions from the action list? you can first reason about your recommendation in less than 50 words. Don't forget to put ### after your reasoning finishes. Then write your suggested actions. \noutput format:\nreason: r\n###\n{output}"
                }
            ]


class GPTRecs_with_fallback(PromptFormat):
    def __init__(self, num_of_recs=5):
        super().__init__(num_of_recs)

        self.num_of_recs = num_of_recs

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, previous_suggestions,
                      posttest=False,
                      prev_exp=None):
        formated_history = self._format_history(history)
        formated_prev_exp = self._format_prev_exp(prev_exp)
        formated_options = self._format_options(valid_subjects, valid_topics, valid_causes, posttest)
        available_actions = self._available_actions(posttest)
        output = "1. choose(cause1)\n2. choose(cause2)" if posttest else "1. ask(x,y)\n2. answer()"
        prompt = [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will recommend actions for the student to do. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}available actions:{available_actions}\n{formated_options}What are your top {2 if posttest else self.num_of_recs} recommended actions from the action list? you can first reason about your recommendation in less than 50 words. Don't forget to put ### after your reasoning finishes. Then write your suggested actions. \noutput format:\nreason: r\n\n###\n{output}"
            }
        ]
        user = f"The student would prefer not to follow your suggestions. Recommend new actions to the student based on the valid actions list. \noutput format:\nreason: r\n\n###\n{output}"
        if len(previous_suggestions) > 0:
            appendix = []

            for i, s in enumerate(previous_suggestions):
                appendix.append({
                    "role": "assistant",
                    "content": f"{s}"
                })
                appendix.append({
                    "role": "user",
                    "content": user
                })
            prompt += appendix

        return prompt


class GPTPlays(PromptFormat):
    def __init__(self):
        super().__init__()

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, posttest=False,
                      prev_exp=None):
        formated_history = self._format_history(history)
        formated_prev_exp = self._format_prev_exp(prev_exp)
        formated_options = self._format_options(valid_subjects, valid_topics, valid_causes, posttest)
        available_actions = self._available_actions(posttest)
        output = "choose(teething)" if posttest else "ask(baby,age)"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure.Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}Interaction History:\n{formated_history}available actions:{available_actions}\n{formated_options}What action would you take next?\noutput format:\nreason: your reason\n$$$\n{output}"
                    # you can first reason about your choice in less than 50 words. Don't forget to put $$$ after your reasoning finishes. Then write your chosen action.
                }
            ]


class CLINChooses(PromptFormat):
    def __init__(self, topk=5):
        super().__init__(topk)
        self.topk = topk

    def format_prompt(self, history, subject, problem, choices, summary, posttest=False, prev_exp=None):
        formated_history = self._format_history(history)
        formated_prev_exp = self._format_prev_exp(prev_exp)
        formated_choices = self._format_choices(choices, self.topk)
        question = "What should the student answer?" if posttest else "What should the student choose to do?"
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                         f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for completing your task:\n"
        user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                        Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale of at most 50 words. Finally, write ### followed by the single next action you would like to take.
                        """
        return [
            {
                "role": "system",
                "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure.Remember that asking the same question again will not give you a different answer. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
            },
            {
                "role": "user",
                "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}{summary_prompt}\n{summary}\nObservation History:\n{formated_history}Student top choices:\n{formated_choices} {question} answer with the choice number.{user_em}\nexample output:\n1, 3\n$$$\nreason: r\n\n###\n1"
            }
        ]


class CLINRecs(PromptFormat):
    def __init__(self, num_of_recs=5):
        super().__init__(num_of_recs)
        self.num_of_recs = num_of_recs

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, summary,
                      posttest=False, prev_exp=None):
        formated_history = self._format_history(history)
        formated_prev_exp = self._format_prev_exp(prev_exp)
        formated_options = self._format_options(valid_subjects, valid_topics, valid_causes, posttest)
        available_actions = self._available_actions(posttest)
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                         f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for completing your task:\n"
        user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                        Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale of at most 50 words. Finally, write ### followed by your list of recommended actions.
                        """
        output = "1. choose(cause1)\n2. choose(cause2)" if posttest else "1. ask(x,y)\n2. answer()"
        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will recommend actions for the student to do. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}{summary_prompt}\n{summary}\nObservation History:\n{formated_history}available actions:{available_actions}\n{formated_options}What are your top {2 if posttest else self.num_of_recs} recommended actions from the action list? {user_em}\noutput format:\n1, 3\n$$$\nreason: r\n###\n{output}"
                }
            ]


class CLINPlays(PromptFormat):
    def __init__(self):
        super().__init__()

    def format_prompt(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, summary,
                      posttest=False, prev_exp=None):
        formated_history = self._format_history(history)
        formated_prev_exp = self._format_prev_exp(prev_exp)
        formated_options = self._format_options(valid_subjects, valid_topics, valid_causes, posttest)
        available_actions = self._available_actions(posttest)
        summary_prompt = f"Here is a summary of learnings based on your previous attempts on this task." \
                         f"These learnings capture important pre-conditions: X MAY BE NECESSARY to Y, X SHOULD BE NECESSARY to Y, and mistakes: X MAY NOT CONTRIBUTE to Y, X DOES NOT CONTRIBUTE to Y. These can be useful for predicting your next action:\n"
        user_em = """First, scan the (unordered) list of learnings, if provided. Decide if any of the learnings are applicable given the last observation to make progress in this task. Then only use selected learnings, if any, to construct a rationale for picking the next action. If no Learning is selected, construct the rationale based on the observation history. Format your response as follows:
                Write the selected learning_ids as a comma separated list; the list can be empty if no learnings selected. Then, write $$$ followed by the rationale of at most 50 words. Finally, write ### followed by the single next action you would like to take.
                """
        output = "choose(teething)" if posttest else "ask(baby,age)"

        return \
            [
                {
                    "role": "system",
                    "content": "You are a pharmacist helping a pharmacy student go through an educational game. You will receive a description of the customer and you will help the student ask a series of questions needed to help the customer. Make sure you are keeping the discussion short with the customer and offer a diagnosis once you are sure. At each step, you will receive the list of available actions for the student and you will choose the best action for the student to do. Always follow the format of available actions. if the patient asks you what's to most probable cause behind their problem you can't continue asking questions make sure you recommend diagnoses then."
                },
                {
                    "role": "user",
                    "content": f"Task: Find the cause behind the {subject}'s {problem}\nYour previous experience with similar tasks:\n{formated_prev_exp}{summary_prompt}\n{summary}\nObservation History:\n{formated_history}available actions:{available_actions}\n{formated_options}What action would you take next? {user_em}\noutput format:\n1, 3\n$$$\nreason: your reason\n***\n{output}"
                }
            ]

    def parse_response(self, response):
        print(response.choices[0].message.content)
        print(response.choices[0].message.content)
        return response.choices[0].message.content


class Chooser_or_Recommender():
    def __init__(self, prompt_format, model="gpt-3.5-turbo", temperature=0,
                 max_tokens=200) -> None:
        self.prompt_format = prompt_format
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def cor(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, choices, summary=None,
            posttest=False, prev_exp=None):

        if summary is None:
            prompt = self.prompt_format.format_prompt(history, subject, problem, valid_subjects, valid_topics,
                                                      valid_causes, choices, posttest=posttest, prev_exp=prev_exp)
        else:
            prompt = self.prompt_format.format_prompt(history, subject, problem, valid_subjects, valid_topics,
                                                      valid_causes, choices, summary, posttest=posttest,
                                                      prev_exp=prev_exp)
        for (i, p) in enumerate(prompt):
            length = len(p["content"].split())
            print(length)
            if length > MAX_LEN:
                prompt[i]["content"] = " ".join(p["content"].split()[:MAX_LEN])
        if "gpt" in self.model:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0)
        elif self.model in ["mixtral-8x7b-32768", "gemma2-9b-it", "llama3-70b-8192", "llama3-8b-8192"]:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = None
            while response is None:
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0)
                except:
                    response = None
        else:
            openai.api_key = "EMPTY"
            openai.base_url = "http://localhost:8000/v1/"
            if prompt[0]["role"] == "system":
                prompt[1]["content"] = prompt[0]["content"] + prompt[1]["content"]
                prompt = prompt[1::]
            ##print(prompt)
            response = openai.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        return self.prompt_format.parse_response(response)


class Chooser(PromptExecutor):
    def choose(self, history, subject, problem, choices, summary=None, posttest=False, prev_exp=None):
        prompt = self.prompt_format.format_prompt(history, subject, problem, choices, posttest=posttest,
                                                  prev_exp=prev_exp)
        for i, p in enumerate(prompt):
            if len(p["content"].split()) > MAX_LEN:
                prompt[i]["content"] = " ".join(p["content"].split()[:MAX_LEN])
        response = self.execute_prompt(prompt)
        return self.prompt_format.parse_response(response)


class Recommender(PromptExecutor):
    def rec(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, summary=None, posttest=False,
            prev_exp=None):
        """
        Formats and sends a recommendation prompt, then processes the response.

        Args:
            history (list): Conversation history.
            subject (str): Main subject of the interaction.
            problem (str): The customer's problem or issue.
            valid_subjects, valid_topics, valid_causes (list): Valid options for each category.
            summary (str, optional): Summary of previous interactions or findings.
            posttest (bool, optional): Indicates if this is a posttest scenario.
            prev_exp (list, optional): Previous experiences to inform the response.
        Returns:
            str: The parsed response from the AI model.
        """

        # Generate prompt using the prompt format specified for recommendations
        prompt = self.prompt_format.format_prompt(
            history, subject, problem, valid_subjects, valid_topics, valid_causes,
            summary=summary, posttest=posttest, prev_exp=prev_exp
        )

        # Limit each prompt segment to MAX_LEN words to avoid model constraints
        for i, p in enumerate(prompt):
            if len(p["content"].split()) > MAX_LEN:
                prompt[i]["content"] = " ".join(p["content"].split()[:MAX_LEN])

        # Execute the prompt and return the parsed response
        response = self.execute_prompt(prompt)
        return self.prompt_format.parse_response(response)


class RecommenderWithRetry(PromptExecutor):
    def rec(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, previous_suggestions,
            summary=None, posttest=False, prev_exp=None):
        """
        Formats and sends a recommendation prompt with retry logic, then processes the response.

        Args:
            history (list): Conversation history.
            subject (str): Main subject of the interaction.
            problem (str): The customer's problem or issue.
            valid_subjects, valid_topics, valid_causes (list): Valid options for each category.
            previous_suggestions (list): List of previous suggestions to avoid repetition.
            summary (str, optional): Summary of previous interactions or findings.
            posttest (bool, optional): Indicates if this is a posttest scenario.
            prev_exp (list, optional): Previous experiences to inform the response.
        Returns:
            str: The parsed response from the AI model.
        """

        # Generate prompt using the prompt format specified for recommendations with retries
        prompt = self.prompt_format.format_prompt(
            history, subject, problem, valid_subjects, valid_topics, valid_causes,
            previous_suggestions=previous_suggestions, summary=summary, posttest=posttest,
            prev_exp=prev_exp
        )

        # Limit each prompt segment to MAX_LEN words to avoid model constraints
        for i, p in enumerate(prompt):
            if len(p["content"].split()) > MAX_LEN:
                prompt[i]["content"] = " ".join(p["content"].split()[:MAX_LEN])

        # Execute the prompt and retry on failure if necessary
        response = self.execute_prompt_with_retry(prompt)
        return self.prompt_format.parse_response(response)

    def execute_prompt_with_retry(self, prompt, max_retries=3):
        """
        Executes an API call with retry logic, retrying on failure up to a specified limit.

        Args:
            prompt (list): Formatted prompt messages for the API.
            max_retries (int): Maximum number of retry attempts on failure.

        Returns:
            Response object from the API.
        """
        attempt = 0
        response = None
        while attempt < max_retries:
            try:
                response = self.execute_prompt(prompt)
                break  # If successful, exit the loop
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}. Retrying...")
        if response is None:
            raise RuntimeError(f"Failed to get response after {max_retries} attempts.")
        return response


class Player(PromptExecutor):
    def play(self, history, subject, problem, valid_subjects, valid_topics, valid_causes, summary=None, posttest=False,
             prev_exp=None):
        """
        Formats and sends a play prompt, then processes the response to determine the next action.

        Args:
            history (list): Conversation history.
            subject (str): Main subject of the interaction.
            problem (str): The customer's problem or issue.
            valid_subjects, valid_topics, valid_causes (list): Valid options for each category.
            summary (str, optional): Summary of previous interactions or findings.
            posttest (bool, optional): Indicates if this is a posttest scenario.
            prev_exp (list, optional): Previous experiences to inform the response.
        Returns:
            str: The parsed response from the AI model, representing the next recommended action.
        """

        # Generate the formatted prompt using the specified prompt format for play scenarios
        prompt = self.prompt_format.format_prompt(
            history, subject, problem, valid_subjects, valid_topics, valid_causes,
            summary=summary, posttest=posttest, prev_exp=prev_exp
        )

        # Limit each prompt segment to MAX_LEN words to meet API constraints
        for i, p in enumerate(prompt):
            if len(p["content"].split()) > MAX_LEN:
                prompt[i]["content"] = " ".join(p["content"].split()[:MAX_LEN])

        # Execute the prompt and retrieve the response
        response = self.execute_prompt(prompt)
        return self.prompt_format.parse_response(response)
