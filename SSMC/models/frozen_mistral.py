from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import yaml
import torch
import torch.nn as nn

section_prompts = {
    "issue_statements": """You are a legal writing expert tasked with creating a single, 
    concise issue statement for a legal brief. Generate a new issue statement based on the case context, 
    following the style and format of the example provided. An effective issue statement should incorporate 
    these elements (DO NOT use numbered lists or bullet points):
    - Use either a 'whether' or 'under [law]' format consistently
    - Identify the controlling law and critical facts that trigger the legal issue
    - Be concise (ideally one sentence that can be read in one breath)
    - Connect legal principles to key facts from the case context
    - Use clear syntax and avoid ambiguous references
    - Present facts objectively while maintaining persuasiveness
    - Avoid including unnecessary or irrelevant facts
    IMPORTANT: Format your response as a concise paragraph, not as a numbered list or bullet points. The example will show the proper format to follow.""",
    
    "rule_statements": """You are a legal writing expert tasked with creating a comprehensive rule statement 
    for a legal brief. Generate a new rule statement based on the case context, following the style and format 
    of the example provided. An effective rule statement should incorporate these elements into cohesive 
    paragraphs (DO NOT use numbered lists or bullet points):
    - Be structured as one of the common types (elements test, balancing test, totality test, or exceptions test)
    - Be grounded in legal authority with appropriate citations
    - Be written in clear and simple language
    - Avoid overly vague terms that might lead to inconsistent interpretation
    - Provide a defined and measurable standard for decision-making
    - Integrate key legal concepts into a single coherent standard rather than listing them separately
    - Present a logical generalization rather than a patchwork of separate statements
    - Avoid unnecessary complexity while maintaining accuracy
    - Be flexible enough to apply across similar cases while maintaining specificity
    IMPORTANT: Format your response as cohesive paragraphs of prose, not as a numbered list or bullet points. The example will show the proper paragraph format to follow.""",
    
    "case_comparisons": """You are a legal writing expert tasked with creating a comprehensive case comparison 
    for a legal brief. Generate a new case comparison based on the case context, following the style and format 
    of the example provided.
    An effective case comparison should incorporate these elements into cohesive paragraphs 
    (DO NOT use numbered lists or bullet points):
    - Clearly explain and illustrate the rule through carefully selected case examples
    - Include a thesis statement that asserts the legal principle the cases illustrate
    - Use appropriate transitions between cases (e.g., "For example," "Similarly," "In contrast")
    - Describe case parties in general terms rather than using citations
    - Include critical facts, holdings, and reasoning from each case
    - Vary the depth of explanation based on case complexity
    - Strike a balance between thoroughness and succinctness
    - Avoid mentioning the client in the rule support section
    - Use explanatory parentheticals that show how each case illustrates the proposition
    IMPORTANT: Format your response as cohesive paragraphs of prose, not as a numbered list or bullet points. The example will show the proper paragraph format to follow.""",
    
    "rule_applications": """You are a legal writing expert tasked with creating a comprehensive rule application 
    for a legal brief. Generate a new rule application based on the case context, following the style and format 
    of the example provided.
    An effective rule application should incorporate the following elements into cohesive paragraphs 
    (DO NOT use numbered lists or bullet points in your response):
    - Clearly articulate the relevant legal rule before applying it
    - Clearly identify whether it is analogizing to or distinguishing from precedent
    - State the legally significant facts of the precedential case
    - Compare legally significant facts from precedent with key facts of the current case
    - Apply the reasoning of precedential cases to support an outcome
    - Maintain parallel structure when presenting facts from both cases
    - Compare multiple cases rather than relying on a single case
    - Acknowledge unfavorable facts rather than ignoring them
    - Provide counterarguments or mitigating factors for unfavorable precedents
    - Conclude with a prediction or argument about how the court should rule
    IMPORTANT: Format your response as cohesive paragraphs of prose, not as a numbered list or bullet points. The example will show the proper paragraph format to follow.""",
    }

case_context = """Ms. Michelle Archer was arrested for disorderly conduct in Massachusetts after an incident in an outdoor parking lot in a major city. The incident occurred following an anti-police brutality protest. An officer told Ms. Archer she could not record a police interaction secretly, and in response, she became agitated and yelled at the officer. As the encounter continued, she got louder.
Archer was observed making loud noises described as "bouncing off the walls" by witness Santos, who stated "he didn't know she could make that much noise." The surrounding buildings in the parking lot acted as an "echoing chamber" which amplified the volume. Archer yelled, "I'm not going to stop; I don't have to. You know this, you are all trained in this. I am sick of police abusing their power in this community."
Archer's interaction with police lasted seven to ten minutes, during which she refused to comply with officers' commands to "calm down and leave the area." The officer warned her that she would be arrested for being disorderly if she did not comply. After a final order to leave, she was arrested. Her behavior drew a crowd of fifteen to twenty people along a fence, some of whom held signs from the earlier protest. Archer made passionate accusations about institutionalized policing with large gestures, getting "in the face" of the arresting officer. This occurred during a time of heightened tensions following the murder of George Floyd and ensuing Black Lives Matter protests.
Under Massachusetts law, which incorporates §250.2(1)(a) and (c) of the Model Penal Code (1980), a person is disorderly if, with the purpose of causing public inconvenience, annoyance or alarm, or recklessly creating a risk thereof, the person engages in fighting, threatening, or in violent or tumultuous behavior, or creates a hazardous or physically offensive condition by an act which serves no legitimate purpose. Commonwealth v. Mulvey, 784 N.E.2d 1138, 1141 (Mass. App. Ct. 2003).
Conduct is tumultuous when it creates extreme noise and commotion that causes a public nuisance. Commonwealth v. Sholley, 739 N.E.2d 236, 242 (Mass. 2000). The noise and commotion created is considered extreme if it goes far beyond the normal level of noise and commotion commonly encountered in the area in which the conduct occurs. Id. at 243. Additionally, refusing a police order is tumultuous if refusing the order exposes both the police and the public to danger by reducing the police officer's ability to maintain order. Commonwealth v. Marcavage, 918 N.E.2d 855, 859 (Mass. App. Ct. 2009).
Relevant precedents include:
- Lopiano (brief agitated expressions not deemed tumultuous; defendant was arrested quickly, making the occurrence a momentary outburst)
- Sholley (defendant ran around courthouse yelling "out of control," drawing people from their chambers; court held that the defendant's running and screaming, which went beyond the court's normal "hurly-burly," was sufficiently extreme)
- Marcavage (defendant refused police orders and resisted confiscation of megaphone, drawing hostile crowd; court held that by refusing police orders, the defendant created a threat to public safety by causing hostility and disrespect towards police)
- Ramos (defendant loudly screamed and flailed arms during fifteen-minute tirade, drawing neighbors from homes; court argued that the volume of speech was extreme enough to constitute tumultuous behavior)
The legal question is whether Archer's conduct meets the definition of "tumultuous" under Massachusetts disorderly conduct law."""


def load_config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class FrozenMistral(nn.Module):
    """
    Still using mistral 7B instruct but this version doesnt have its weight updated
    loaded with 8-bit quantization because of space issues
    As i understand mlps still learn in full precsion so loading the model in 8-bit doesnt matter
    it's frozen so numerical precision in regards to weights being updated is a moot point.
    """
    def __init__(self, device=None):
        super().__init__()
        config = load_config()
        model = config["model"]
    
        self.tokenizer = AutoTokenizer.from_pretrained(model["name"])
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model["name"], 
            torch_dtype = model["dtype"], 
            use_auth_token = True,
            trust_remote_code = model["trust_remote_code"],
            use_cache=False,
            device_map="auto",
            quantization_config=quantization_config
        )
        
        for param in self.model.parameters():
            param.requires_grad = False
    
        self.model.eval()

    def forward(self, soft_prompt):
        """Call the forward pass on the soft_prompts, which are the embeddings (from paper)"""
        gpu = next(self.model.parameters()).device
        soft_prompt = soft_prompt.to(gpu, dtype=next(self.model.parameters()).dtype)
        outputs = self.model(inputs_embeds=soft_prompt, return_dict=True)
        return outputs
    
    def generate(self, soft_prompt, section, input_text=None):
        """Generates synthetic data based on the learned embeddings for a section"""
        gpu = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        soft_prompt = soft_prompt.to(gpu, dtype=dtype)
        
        config = load_config()
        generation = config["generation"].copy()
        
        # copy the section config and get the correct token generation limit
        if section and section in generation["max_new_tokens"]:
            max_tokens = generation["max_new_tokens"][section]
            gen_config = {k: v for k, v in generation.items() if k != "max_new_tokens"}
            gen_config["max_new_tokens"] = max_tokens
            # Assemble the prompt with: instruction for the section, case_context, and an example to imitate
            if input_text:
                prompt_template = section_prompts.get(section)
                prompt = f"""INSTRUCTIONS:
                {prompt_template}
                CASE CONTEXT:
                {case_context}
                EXAMPLE TO FOLLOW:
                {input_text}
                IMPORTANT: Your response should CLOSELY MIMIC the example above in terms of style, structure, tone, reasoning approach, and level of detail. Follow the same pattern of analysis, use similar paragraph organization, and maintain a comparable length. While addressing the specific facts of the case context, your writing style should be indistinguishable from the example.
                TASK: Generate a new {section.rstrip('s')} for the case context above that follows the style and format of the example. Do not repeat any part of the instructions, case context, or example in your response.
                ### YOUR RESPONSE BELOW ###
                """
                input_tokens = self.tokenizer(prompt, return_tensors="pt").to(gpu)
                with torch.no_grad(): # Convert token IDs to embedding vectors
                    input_embeds = self.model.get_input_embeddings()(input_tokens.input_ids)
                # add the soft prompt dense vector to the input embeddings
                combined_embeds = torch.cat([soft_prompt, input_embeds], dim=1)
                outputs = self.model.generate(inputs_embeds=combined_embeds, **gen_config)
            else:
                outputs = self.model.generate(inputs_embeds=soft_prompt, **gen_config)
            return outputs
        else:
            raise ValueError(f"Invalid section: {section}")

    '''
    Decode the final output, and only return the synthetic data after the delimiter
    '''
    def decode(self, tokens):
        decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        separator = "### YOUR RESPONSE BELOW ###"
        if separator in decoded_text:
            return decoded_text.split(separator)[1].strip()
        return decoded_text