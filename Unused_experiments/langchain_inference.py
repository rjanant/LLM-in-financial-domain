# %%
#inference - llama2_chat_7b and llama3_8b

from transformers import AutoTokenizer,AutoModelForCausalLM
import torch,os

model_name = "meta-llama/Llama-2-7b-chat-hf" 
output_merged_dir = "3006_final_llama2/final_merged_checkpoint"
model = AutoModelForCausalLM.from_pretrained(output_merged_dir, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%
#llama2_model: 

inference_template = """<s>[INST]
<<SYS>>
You are a helpful AI assistant specializing in financial analysis. Answer the following question based on the given context.
<</SYS>>
{user_message} [/INST]
"""

def create_prompt_formats_infer(sample):
    context = sample["context"]
    question = sample["question"]
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    text = inference_template.format(user_message=user_message)
    return {"text": text}

samples = [
    {
        "context": "On a sequential basis, ADG revenues were up 3.3%, driven by an increase in volumes of approximately 8%, partially offset by a decrease in average selling prices of approximately 5%, mostly attributable to product mix. AMS revenues increased 12.1% driven by Analog and Imaging products. AMS increase was due to an increase of approximately 5% in average selling prices, entirely due to product mix, and to higher volumes of approximately of 7%. MDG revenues increased by 7.9%, mainly driven by Microcontrollers, due to both higher average selling prices of approximately 6%, entirely due to product mix, and higher volumes of approximately 2%. On a year-over-year basis, fourth quarter net revenues increased by 4.0%. ADG revenues decreased 4.5% compared to the year-ago quarter on lower revenues in both Automotive and Power Discrete. The decrease was entirely due to lower average selling prices of approximately 4%, while volumes remained substantially flat. The decrease in average selling prices was a combination of less favorable product mix and lower selling prices. AMS fourth quarter revenues grew 9.9% year-over-year, mainly driven by Analog and Imaging. The increase was entirely due to higher average selling prices of approximately 18%, entirely attributable to product mix,\npartially offset by lower volumes of approximately 8%. MDG fourth quarter revenues increased by 7.6%, mainly driven by Microcontrollers. The increase was due to higher average selling prices of approximately 9%,\nentirely due to improved product mix.\nTable:\n\t\tThree Months Ended\t\t% Variation\t\n\tDecember 31, 2019\tSeptember 29, 2019\tDecember 31, 2018\tSequential\tYear-Over-Year\n\t\t\t(Unaudited, in millions)\t\t\nAutomotive and Discrete Group (ADG)\t$924\t$894\t$967\t3.3%\t(4.5)%\nAnalog, MEMS and Sensors Group (AMS)\t1,085\t968\t988\t12.1\t9.9\nMicrocontrollers and Digital ICs Group (MDG)\t742\t688\t689\t7.9\t7.6\nOthers\t3\t3\t4\t\u2014\t\u2014\nTotal consolidated net revenues\t$2,754\t$2,553\t$2,648\t7.9%\t4.0%",
        "question": "What is the average net revenues from Analog, MEMS and Sensors Group (AMS) for the period December 31, 2019 and 2018?",
        "answer": '' #"1036.5"
    },
    {
        "context": "14 Taxation UK corporation tax for the year-ended 31 March 2019 is calculated at 19% (2018: 19%) of the estimated assessable loss for the period.\nTable:\n\tYear-ended 31 March 2019\tYear-ended 31 March 2018 Restated See note 2\n\t$M\t$M\nCurrent income tax:\t\t\nUK corporation tax\t1.3\t1.2\nAdjustments in respect of previous years UK tax\t0.3\t0.3\nOverseas tax before exceptional items\t22.2\t23.0\nAdjustment in respect of previous years\t13.1\t10.2\nTotal current tax charge\t36.9\t34.7\nDeferred tax:\t\t\nOrigination and reversal of temporary differences\t2.5\t(16.7)\nImpact of changes in US tax rate\t-\t5.4\nAdjustment in respect of previous years\t(12.7)\t(3.5)\nTotal deferred tax credit\t(10.2)\t(14.8)\nTotal income tax charge\t26.7\t19.9",
        "question": "What was UK corporation tax for the year-ended 31 March 2019 calculated as?",
        "answer": '' #"19% (2018: 19%) of the estimated assessable loss for the period."
    },
    {
        "context": "consumer foods net sales increased $ 303 million , or 5% ( 5 % ) , for the year to $ 6.8 billion . results reflect an increase of three percentage points from improved net pricing and product mix and two percentage points of improvement from higher volumes . net pricing and volume improvements were achieved in many of the company 2019s priority investment and enabler brands . the impact of product recalls partially offset these improvements . the company implemented significant price increases for many consumer foods products during the fourth quarter of fiscal 2008 . continued net sales improvements are expected into fiscal 2009 when the company expects to receive the benefit of these pricing actions for full fiscal periods . sales of some of the company 2019s most significant brands , including chef boyardee ae , david ae , egg beaters ae , healthy choice ae , hebrew national ae , hunt 2019s ae , marie callender 2019s ae , manwich ae , orville redenbacher 2019s ae , pam ae , ro*tel ae , rosarita ae , snack pack ae , swiss miss ae , wesson ae , and wolf ae grew in fiscal 2008 . sales of act ii ae , andy capp ae , banquet ae , crunch 2018n munch ae , kid cuisine ae , parkay ae , pemmican ae , reddi-wip ae , and slim jim ae declined in fiscal 2008 . net sales in the consumer foods segment are not comparable across periods due to a variety of factors . the company initiated a peanut butter recall in the third quarter of fiscal 2007 and reintroduced peter pan ae peanut butter products in august 2007 . sales of all peanut butter products , including both branded and private label , in fiscal 2008 were $ 14 million lower than comparable amounts in fiscal 2007 . consumer foods net sales were also adversely impacted by the recall of banquet ae and private label pot pies in the second quarter of fiscal 2008 . net sales of pot pies were lower by approximately $ 22 million in fiscal 2008 , relative to fiscal 2007 , primarily due to product returns and lost sales of banquet ae and private label pot pies . sales from alexia foods and lincoln snacks , businesses acquired in fiscal 2008 , totaled $ 66 million in fiscal 2008 . the company divested a refrigerated pizza business during the first half of fiscal 2007 . sales from this business were $ 17 million in fiscal food and ingredients net sales were $ 4.1 billion in fiscal 2008 , an increase of $ 706 million , or 21% ( 21 % ) . increased sales are reflective of higher sales prices in the company 2019s milling operations due to higher grain prices , and price and volume increases in the company 2019s potato and dehydrated vegetable operations . the fiscal 2007 divestiture of an oat milling operation resulted in a reduction of sales of $ 27 million for fiscal 2008 , partially offset by increased sales of $ 18 million from the acquisition of watts brothers in february 2008 . international foods net sales increased $ 65 million to $ 678 million . the strengthening of foreign currencies relative to the u.s . dollar accounted for approximately $ 36 million of this increase . the segment achieved a 5% ( 5 % ) increase in sales volume in fiscal 2008 , primarily reflecting increased unit sales in canada and mexico , and modest increases in net pricing . gross profit ( net sales less cost of goods sold ) ( $ in millions ) reporting segment fiscal 2008 gross profit fiscal 2007 gross profit % (  % ) increase/ ( decrease ) . <table class='wikitable'><tr><td>1</td><td>reporting segment</td><td>fiscal 2008 gross profit</td><td>fiscal 2007 gross profit</td><td>% (  % ) increase/ ( decrease )</td></tr><tr><td>2</td><td>consumer foods</td><td>$ 1802</td><td>$ 1923</td><td>( 6 ) % (  % )</td></tr><tr><td>3</td><td>food and ingredients</td><td>724</td><td>590</td><td>23% ( 23 % )</td></tr><tr><td>4</td><td>international foods</td><td>190</td><td>180</td><td>6% ( 6 % )</td></tr><tr><td>5</td><td>total</td><td>$ 2716</td><td>$ 2693</td><td>1% ( 1 % )</td></tr></table> the company 2019s gross profit for fiscal 2008 was $ 2.7 billion , an increase of $ 23 million , or 1% ( 1 % ) , over the prior year . the increase in gross profit was largely driven by results in the food and ingredients segment , reflecting higher margins in the company 2019s milling and specialty potato operations , largely offset by reduced gross profits in the consumer foods segment . costs of implementing the company 2019s restructuring plans reduced gross profit by $ 4 million and $ 46 million in fiscal 2008 and fiscal 2007 , respectively. .\nQuestion: what is the gross profit generated by consumer foods in 2007?\nAnswer: 1923.0\nQuestion: what about the total gross profit?\nAnswer: 2693.0\nQuestion: what proportion is coming from consumer foods?\nAnswer: 0.71407\nQuestion: what about the gross profit generated by consumer foods in 2008?\nAnswer: 1802.0\nQuestion: and total gross profit in 2008?\nAnswer: 2716.0\nQuestion: what portion does this represent in 2008?\n",
        "question": "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided.\nAnswer the last question in the given context, if history is provided for previous questions-answers use that information (if required based on the context) to get the answer for the last question",
        "answer": '' #"0.66348"
    },
    {
        "context": "devon energy corporation and subsidiaries notes to consolidated financial statements 2014 ( continued ) the following methods and assumptions were used to estimate the fair values in the tables above . fixed-income securities 2014 devon 2019s fixed-income securities consist of u.s . treasury obligations , bonds issued by investment-grade companies from diverse industries , and asset-backed securities . these fixed-income securities are actively traded securities that can be redeemed upon demand . the fair values of these level 1 securities are based upon quoted market prices . devon 2019s fixed income securities also include commingled funds that primarily invest in long-term bonds and u.s . treasury securities . these fixed income securities can be redeemed on demand but are not actively traded . the fair values of these level 2 securities are based upon the net asset values provided by the investment managers . equity securities 2014 devon 2019s equity securities include a commingled global equity fund that invests in large , mid and small capitalization stocks across the world 2019s developed and emerging markets . these equity securities can be redeemed on demand but are not actively traded . the fair values of these level 2 securities are based upon the net asset values provided by the investment managers . at december 31 , 2010 , devon 2019s equity securities consisted of investments in u.s . large and small capitalization companies and international large capitalization companies . these equity securities were actively traded securities that could be redeemed upon demand . the fair values of these level 1 securities are based upon quoted market prices . at december 31 , 2010 , devon 2019s equity securities also included a commingled fund that invested in large capitalization companies . these equity securities could be redeemed on demand but were not actively traded . the fair values of these level 2 securities are based upon the net asset values provided by the investment managers . other securities 2014 devon 2019s other securities include commingled , short-term investment funds . these securities can be redeemed on demand but are not actively traded . the fair values of these level 2 securities are based upon the net asset values provided by investment managers . devon 2019s hedge fund and alternative investments include an investment in an actively traded global mutual fund that focuses on alternative investment strategies and a hedge fund of funds that invests both long and short using a variety of investment strategies . devon 2019s hedge fund of funds is not actively traded and devon is subject to redemption restrictions with regards to this investment . the fair value of this level 3 investment represents the fair value as determined by the hedge fund manager . included below is a summary of the changes in devon 2019s level 3 plan assets ( in millions ) . .\n\n.\n\nTable:\ndecember 31 2009\t$ 51\npurchases\t3\ninvestment returns\t4\ndecember 31 2010\t58\npurchases\t33\ninvestment returns\t-1 ( 1 )\ndecember 31 2011\t$ 90",
        "question": "what was the percentage change in devon 2019s level 3 plan assets from 2009 to 2010",
        "answer": '' #"13.7%"
    },
    {
        "context": "the goldman sachs group , inc . and subsidiaries notes to consolidated financial statements commercial lending . the firm 2019s commercial lending commitments are extended to investment-grade and non-investment-grade corporate borrowers . commitments to investment-grade corporate borrowers are principally used for operating liquidity and general corporate purposes . the firm also extends lending commitments in connection with contingent acquisition financing and other types of corporate lending , as well as commercial real estate financing . commitments that are extended for contingent acquisition financing are often intended to be short-term in nature , as borrowers often seek to replace them with other funding sources . sumitomo mitsui financial group , inc . ( smfg ) provides the firm with credit loss protection on certain approved loan commitments ( primarily investment-grade commercial lending commitments ) . the notional amount of such loan commitments was $ 25.70 billion and $ 26.88 billion as of december 2017 and december 2016 , respectively . the credit loss protection on loan commitments provided by smfg is generally limited to 95% ( 95 % ) of the first loss the firm realizes on such commitments , up to a maximum of approximately $ 950 million . in addition , subject to the satisfaction of certain conditions , upon the firm 2019s request , smfg will provide protection for 70% ( 70 % ) of additional losses on such commitments , up to a maximum of $ 1.13 billion , of which $ 550 million and $ 768 million of protection had been provided as of december 2017 and december 2016 , respectively . the firm also uses other financial instruments to mitigate credit risks related to certain commitments not covered by smfg . these instruments primarily include credit default swaps that reference the same or similar underlying instrument or entity , or credit default swaps that reference a market index . warehouse financing . the firm provides financing to clients who warehouse financial assets . these arrangements are secured by the warehoused assets , primarily consisting of retail and corporate loans . contingent and forward starting collateralized agreements / forward starting collateralized financings contingent and forward starting collateralized agreements includes resale and securities borrowing agreements , and forward starting collateralized financings includes repurchase and secured lending agreements that settle at a future date , generally within three business days . the firm also enters into commitments to provide contingent financing to its clients and counterparties through resale agreements . the firm 2019s funding of these commitments depends on the satisfaction of all contractual conditions to the resale agreement and these commitments can expire unused . letters of credit the firm has commitments under letters of credit issued by various banks which the firm provides to counterparties in lieu of securities or cash to satisfy various collateral and margin deposit requirements . investment commitments investment commitments includes commitments to invest in private equity , real estate and other assets directly and through funds that the firm raises and manages . investment commitments included $ 2.09 billion and $ 2.10 billion as of december 2017 and december 2016 , respectively , related to commitments to invest in funds managed by the firm . if these commitments are called , they would be funded at market value on the date of investment . leases the firm has contractual obligations under long-term noncancelable lease agreements for office space expiring on various dates through 2069 . certain agreements are subject to periodic escalation provisions for increases in real estate taxes and other charges . the table below presents future minimum rental payments , net of minimum sublease rentals . $ in millions december 2017 .\n\nrent charged to operating expenses was $ 273 million for 2017 , $ 244 million for 2016 and $ 249 million for 2015 . goldman sachs 2017 form 10-k 163 .\n\nTable:\n$ in millions\tas of december 2017\n2018\t$ 299\n2019\t282\n2020\t262\n2021\t205\n2022\t145\n2023 - thereafter\t771\ntotal\t$ 1964",
        "question": "minimum rents due in 2023 - thereafter are what percent of total future minimum rental payments?\\\\n",
        "answer": '' #"39.3%"
    },
    {
        "context": "EASTERN China-based IT company Inspur will establish a fully controlled subsidiary in the British Virgin Islands and another in the United States, with each investment not exceeding $US10 million ($A13.60 million), according to a filing with the Shenzhen Stock Exchange on Saturday, the Xinhua news agency reported.",
        "question": "Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of \"relation1: word1, word2; relation2: word3, word4\". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.",
        "answer": '' #"stock_exchange: Inspur, Shenzhen Stock Exchange"
    }
]
for sample in samples:
    formatted_sample = create_prompt_formats_infer(sample)

    inputs = tokenizer(
        [formatted_sample['text']],
        return_tensors="pt",
        truncation=True,
        max_length=4096  
    ).to("cuda")

    outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=1.0,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

    generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    predicted_value = generated_output.split("[/INST]")[-1].strip()

    print(f"Context: {sample['context'][:100]}...")  
    print(f"Question: {sample['question']}")
    print(f"Predicted Answer: {predicted_value}\n")
    print("-" * 50)  # Separator for readability

# %%
few_shot_examples = [
   {
        "context": "On a sequential basis, ADG revenues were up 3.3%, driven by an increase in volumes of approximately 8%, partially offset by a decrease in average selling prices of approximately 5%, mostly attributable to product mix. AMS revenues increased 12.1% driven by Analog and Imaging products. AMS increase was due to an increase of approximately 5% in average selling prices, entirely due to product mix, and to higher volumes of approximately of 7%. MDG revenues increased by 7.9%, mainly driven by Microcontrollers, due to both higher average selling prices of approximately 6%, entirely due to product mix, and higher volumes of approximately 2%. On a year-over-year basis, fourth quarter net revenues increased by 4.0%. ADG revenues decreased 4.5% compared to the year-ago quarter on lower revenues in both Automotive and Power Discrete. The decrease was entirely due to lower average selling prices of approximately 4%, while volumes remained substantially flat. The decrease in average selling prices was a combination of less favorable product mix and lower selling prices. AMS fourth quarter revenues grew 9.9% year-over-year, mainly driven by Analog and Imaging. The increase was entirely due to higher average selling prices of approximately 18%, entirely attributable to product mix,\npartially offset by lower volumes of approximately 8%. MDG fourth quarter revenues increased by 7.6%, mainly driven by Microcontrollers. The increase was due to higher average selling prices of approximately 9%,\nentirely due to improved product mix.\nTable:\n\t\tThree Months Ended\t\t% Variation\t\n\tDecember 31, 2019\tSeptember 29, 2019\tDecember 31, 2018\tSequential\tYear-Over-Year\n\t\t\t(Unaudited, in millions)\t\t\nAutomotive and Discrete Group (ADG)\t$924\t$894\t$967\t3.3%\t(4.5)%\nAnalog, MEMS and Sensors Group (AMS)\t1,085\t968\t988\t12.1\t9.9\nMicrocontrollers and Digital ICs Group (MDG)\t742\t688\t689\t7.9\t7.6\nOthers\t3\t3\t4\t\u2014\t\u2014\nTotal consolidated net revenues\t$2,754\t$2,553\t$2,648\t7.9%\t4.0%",
        "question": "What is the average net revenues from Analog, MEMS and Sensors Group (AMS) for the period December 31, 2019 and 2018?",
        "answer": "1036.5"
    },
    {
        "context": "EASTERN China-based IT company Inspur will establish a fully controlled subsidiary in the British Virgin Islands and another in the United States, with each investment not exceeding $US10 million ($A13.60 million), according to a filing with the Shenzhen Stock Exchange on Saturday, the Xinhua news agency reported.",
        "question": "Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of \"relation1: word1, word2; relation2: word3, word4\". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.",
        "answer": "stock_exchange: Inspur, Shenzhen Stock Exchange"
    },
    {
        "context": "consumer foods net sales increased $ 303 million , or 5% ( 5 % ) , for the year to $ 6.8 billion . results reflect an increase of three percentage points from improved net pricing and product mix and two percentage points of improvement from higher volumes . net pricing and volume improvements were achieved in many of the company 2019s priority investment and enabler brands . the impact of product recalls partially offset these improvements . the company implemented significant price increases for many consumer foods products during the fourth quarter of fiscal 2008 . continued net sales improvements are expected into fiscal 2009 when the company expects to receive the benefit of these pricing actions for full fiscal periods . sales of some of the company 2019s most significant brands , including chef boyardee ae , david ae , egg beaters ae , healthy choice ae , hebrew national ae , hunt 2019s ae , marie callender 2019s ae , manwich ae , orville redenbacher 2019s ae , pam ae , ro*tel ae , rosarita ae , snack pack ae , swiss miss ae , wesson ae , and wolf ae grew in fiscal 2008 . sales of act ii ae , andy capp ae , banquet ae , crunch 2018n munch ae , kid cuisine ae , parkay ae , pemmican ae , reddi-wip ae , and slim jim ae declined in fiscal 2008 . net sales in the consumer foods segment are not comparable across periods due to a variety of factors . the company initiated a peanut butter recall in the third quarter of fiscal 2007 and reintroduced peter pan ae peanut butter products in august 2007 . sales of all peanut butter products , including both branded and private label , in fiscal 2008 were $ 14 million lower than comparable amounts in fiscal 2007 . consumer foods net sales were also adversely impacted by the recall of banquet ae and private label pot pies in the second quarter of fiscal 2008 . net sales of pot pies were lower by approximately $ 22 million in fiscal 2008 , relative to fiscal 2007 , primarily due to product returns and lost sales of banquet ae and private label pot pies . sales from alexia foods and lincoln snacks , businesses acquired in fiscal 2008 , totaled $ 66 million in fiscal 2008 . the company divested a refrigerated pizza business during the first half of fiscal 2007 . sales from this business were $ 17 million in fiscal food and ingredients net sales were $ 4.1 billion in fiscal 2008 , an increase of $ 706 million , or 21% ( 21 % ) . increased sales are reflective of higher sales prices in the company 2019s milling operations due to higher grain prices , and price and volume increases in the company 2019s potato and dehydrated vegetable operations . the fiscal 2007 divestiture of an oat milling operation resulted in a reduction of sales of $ 27 million for fiscal 2008 , partially offset by increased sales of $ 18 million from the acquisition of watts brothers in february 2008 . international foods net sales increased $ 65 million to $ 678 million . the strengthening of foreign currencies relative to the u.s . dollar accounted for approximately $ 36 million of this increase . the segment achieved a 5% ( 5 % ) increase in sales volume in fiscal 2008 , primarily reflecting increased unit sales in canada and mexico , and modest increases in net pricing . gross profit ( net sales less cost of goods sold ) ( $ in millions ) reporting segment fiscal 2008 gross profit fiscal 2007 gross profit % (  % ) increase/ ( decrease ) . <table class='wikitable'><tr><td>1</td><td>reporting segment</td><td>fiscal 2008 gross profit</td><td>fiscal 2007 gross profit</td><td>% (  % ) increase/ ( decrease )</td></tr><tr><td>2</td><td>consumer foods</td><td>$ 1802</td><td>$ 1923</td><td>( 6 ) % (  % )</td></tr><tr><td>3</td><td>food and ingredients</td><td>724</td><td>590</td><td>23% ( 23 % )</td></tr><tr><td>4</td><td>international foods</td><td>190</td><td>180</td><td>6% ( 6 % )</td></tr><tr><td>5</td><td>total</td><td>$ 2716</td><td>$ 2693</td><td>1% ( 1 % )</td></tr></table> the company 2019s gross profit for fiscal 2008 was $ 2.7 billion , an increase of $ 23 million , or 1% ( 1 % ) , over the prior year . the increase in gross profit was largely driven by results in the food and ingredients segment , reflecting higher margins in the company 2019s milling and specialty potato operations , largely offset by reduced gross profits in the consumer foods segment . costs of implementing the company 2019s restructuring plans reduced gross profit by $ 4 million and $ 46 million in fiscal 2008 and fiscal 2007 , respectively. .\nQuestion: what is the gross profit generated by consumer foods in 2007?\nAnswer: 1923.0\nQuestion: what about the total gross profit?\nAnswer: 2693.0\nQuestion: what proportion is coming from consumer foods?\nAnswer: 0.71407\nQuestion: what about the gross profit generated by consumer foods in 2008?\nAnswer: 1802.0\nQuestion: and total gross profit in 2008?\nAnswer: 2716.0\nQuestion: what portion does this represent in 2008?\n",
        "question": "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided.\nAnswer the last question in the given context, if history is provided for previous questions-answers use that information (if required based on the context) to get the answer for the last question",
        "answer": "0.66348"
    },
    {
        "context": "14 Taxation UK corporation tax for the year-ended 31 March 2019 is calculated at 19% (2018: 19%) of the estimated assessable loss for the period.\nTable:\n\tYear-ended 31 March 2019\tYear-ended 31 March 2018 Restated See note 2\n\t$M\t$M\nCurrent income tax:\t\t\nUK corporation tax\t1.3\t1.2\nAdjustments in respect of previous years UK tax\t0.3\t0.3\nOverseas tax before exceptional items\t22.2\t23.0\nAdjustment in respect of previous years\t13.1\t10.2\nTotal current tax charge\t36.9\t34.7\nDeferred tax:\t\t\nOrigination and reversal of temporary differences\t2.5\t(16.7)\nImpact of changes in US tax rate\t-\t5.4\nAdjustment in respect of previous years\t(12.7)\t(3.5)\nTotal deferred tax credit\t(10.2)\t(14.8)\nTotal income tax charge\t26.7\t19.9",
        "question": "What was UK corporation tax for the year-ended 31 March 2019 calculated as?",
        "answer": "19% (2018: 19%) of the estimated assessable loss for the period."
    }
]


# %%
inference_template = """<s>[INST]
<<SYS>>
You are a helpful AI assistant specializing in financial analysis. Study the following examples carefully, then answer the new question using a similar approach but with information specific to the new context and question.
<</SYS>>

{user_message} [/INST]
"""
def create_prompt_formats_infer(sample, few_shot_examples):
    # Include a limited number of few-shot examples in the prompt
    example_texts = []
    for i, example in enumerate(few_shot_examples[:2]):  # Limit to 2 examples
        example_text = f"Example {i+1}:\nContext: {example['context']}\nQuestion: {example['question']}\nAnswer: {example['answer']}\n\n"
        example_texts.append(example_text)
    
    # Current question context and question
    current_question_text = f"Now, please answer the following NEW question:\nContext: {sample['context']}\nQuestion: {sample['question']}\nYour answer:"
    
    # Combine few-shot examples with the current question
    full_prompt = inference_template.format(user_message="".join(example_texts) + current_question_text)
    return {"text": full_prompt}


# %%
samples = [
    {
        "context": "Starbucks Corp. SBUX, +2.07% added 0.6% in thin premarket trade after the coffee chain struck a deal to push into sub-Saharan Africa next year .",
        "question": "Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of \"relation1: word1, word2; relation2: word3, word4\". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.",
        "answer": ''
    },
    {
        "context": "The company noted that the commissioning of the new cement plant is part of its Africa expansion strategy and will be the fourth in the series after Ethiopia, Zambia and Cameroun, adding that cement plants due for commissioning this year include that of Senegal and South Africa, while construction works are ongoing in several other African countries.",
        "question": "Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of \"relation1: word1, word2; relation2: word3, word4\". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.",
        "answer": ''
    },
    {
        "context": "vertex pharmaceuticals incorporated notes to consolidated financial statements ( continued ) d . stock-based compensation ( continued ) the company uses the black-scholes valuation model to estimate the fair value of stock options at the grant date . the black-scholes valuation model uses the option exercise price as well as estimates and assumptions related to the expected price volatility of the company 2019s stock , the period during which the options will be outstanding , the rate of return on risk-free investments , and the expected dividend yield for the company 2019s stock to estimate the fair value of a stock option on the grant date . the company validates its estimates and assumptions through consultations with independent third parties having relevant expertise . the fair value of each option granted under the stock and option plans during 2006 was estimated on the date of grant using the black- scholes option pricing model with the following weighted average assumptions : the weighted-average valuation assumptions were determined as follows : 2022 expected stock price volatility : in 2006 , the company changed its method of estimating expected volatility from relying exclusively on historical volatility to relying exclusively on implied volatility . options to purchase the company 2019s stock with remaining terms of greater than one year are regularly traded in the market . expected stock price volatility is calculated using the trailing one month average of daily implied volatilities prior to grant date . 2022 risk-free interest rate : the company bases the risk-free interest rate on the interest rate payable on u.s . treasury securities in effect at the time of grant for a period that is commensurate with the assumed expected option term . 2022 expected term of options : the expected term of options represents the period of time options are expected to be outstanding . the company uses historical data to estimate employee exercise and post-vest termination behavior . the company believes that all groups of employees exhibit similar exercise and post-vest termination behavior and therefore does not stratify employees into multiple groups in determining the expected term of options . 2022 expected annual dividends : the estimate for annual dividends is $ 0.00 , because the company has not historically paid , and does not intend for the foreseeable future to pay , a dividend . expected stock price volatility 57.10 % (  % ) risk-free interest rate 4.74 % (  % ) expected term 5.64 years .\n\n.\n\nTable:\n\t2006\nexpected stock price volatility\t57.10% ( 57.10 % )\nrisk-free interest rate\t4.74% ( 4.74 % )\nexpected term\t5.64 years\nexpected annual dividends\t2014",
        "question": "in 2006 what was the ratio expected stock price volatility to risk-free interest rate",
        "answer": ''
    },
    {
        "context": "item 7a . quantitative and qualitative disclosures about market risk ( amounts in millions ) in the normal course of business , we are exposed to market risks related to interest rates , foreign currency rates and certain balance sheet items . from time to time , we use derivative instruments , pursuant to established guidelines and policies , to manage some portion of these risks . derivative instruments utilized in our hedging activities are viewed as risk management tools and are not used for trading or speculative purposes . interest rates our exposure to market risk for changes in interest rates relates primarily to the fair market value and cash flows of our debt obligations . the majority of our debt ( approximately 89% ( 89 % ) and 93% ( 93 % ) as of december 31 , 2013 and 2012 , respectively ) bears interest at fixed rates . we do have debt with variable interest rates , but a 10% ( 10 % ) increase or decrease in interest rates would not be material to our interest expense or cash flows . the fair market value of our debt is sensitive to changes in interest rates , and the impact of a 10% ( 10 % ) change in interest rates is summarized below . increase/ ( decrease ) in fair market value as of december 31 , 10% ( 10 % ) increase in interest rates 10% ( 10 % ) decrease in interest rates . <table class='wikitable'><tr><td>1</td><td>as of december 31,</td><td>increase/ ( decrease ) in fair market value 10% ( 10 % ) increasein interest rates</td><td>increase/ ( decrease ) in fair market value 10% ( 10 % ) decreasein interest rates</td></tr><tr><td>2</td><td>2013</td><td>$ -26.9 ( 26.9 )</td><td>$ 27.9</td></tr><tr><td>3</td><td>2012</td><td>-27.5 ( 27.5 )</td><td>28.4</td></tr></table> we have used interest rate swaps for risk management purposes to manage our exposure to changes in interest rates . we do not have any interest rate swaps outstanding as of december 31 , 2013 . we had $ 1642.1 of cash , cash equivalents and marketable securities as of december 31 , 2013 that we generally invest in conservative , short-term bank deposits or securities . the interest income generated from these investments is subject to both domestic and foreign interest rate movements . during 2013 and 2012 , we had interest income of $ 24.7 and $ 29.5 , respectively . based on our 2013 results , a 100-basis-point increase or decrease in interest rates would affect our interest income by approximately $ 16.4 , assuming that all cash , cash equivalents and marketable securities are impacted in the same manner and balances remain constant from year-end 2013 levels . foreign currency rates we are subject to translation and transaction risks related to changes in foreign currency exchange rates . since we report revenues and expenses in u.s . dollars , changes in exchange rates may either positively or negatively affect our consolidated revenues and expenses ( as expressed in u.s . dollars ) from foreign operations . the primary foreign currencies that impacted our results during 2013 were the australian dollar , brazilian real , euro , japanese yen and the south african rand . based on 2013 exchange rates and operating results , if the u.s . dollar were to strengthen or weaken by 10% ( 10 % ) , we currently estimate operating income would decrease or increase between 3% ( 3 % ) and 4% ( 4 % ) , assuming that all currencies are impacted in the same manner and our international revenue and expenses remain constant at 2013 levels . the functional currency of our foreign operations is generally their respective local currency . assets and liabilities are translated at the exchange rates in effect at the balance sheet date , and revenues and expenses are translated at the average exchange rates during the period presented . the resulting translation adjustments are recorded as a component of accumulated other comprehensive loss , net of tax , in the stockholders 2019 equity section of our consolidated balance sheets . our foreign subsidiaries generally collect revenues and pay expenses in their functional currency , mitigating transaction risk . however , certain subsidiaries may enter into transactions in currencies other than their functional currency . assets and liabilities denominated in currencies other than the functional currency are susceptible to movements in foreign currency until final settlement . currency transaction gains or losses primarily arising from transactions in currencies other than the functional currency are included in office and general expenses . we have not entered into a material amount of foreign currency forward exchange contracts or other derivative financial instruments to hedge the effects of potential adverse fluctuations in foreign currency exchange rates. .\nQuestion: what was the interest income in 2013?\nAnswer: 24.7\nQuestion: and what was it in 2012?\nAnswer: 29.5\nQuestion: what was, then, the change in interest income over the year?\nAnswer: -4.8\nQuestion: what was the total interest income in 2012?\n",
        "question": "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided.\nAnswer the last question in the given context, if history is provided for previous questions-answers use that information (if required based on the context) to get the answer for the last question",
        "answer": ''
    },
    {
        "context": "A reconciliation of the beginning and ending balance of unrecognized tax benefits is as follows: Included in the total unrecognized tax benefits at December 31, 2019, 2018 and 2017 is $2.4 billion, $2.3 billion and $1.9 billion, respectively, that if recognized, would favorably affect the effective income tax rate.\nTable:\n\t\t\t(dollars in millions)\n\t2019\t2018\t2017\nBalance at January 1,\t$ 2,871\t$2,355\t$ 1,902\nAdditions based on tax positions related to the current year\t149\t160\t219\nAdditions for tax positions of prior years\t297\t699\t756\nReductions for tax positions of prior years\t(300)\t(248)\t(419)\nSettlements\t(58)\t(40)\t(42)\nLapses of statutes of limitations\t(89)\t(55)\t(61)\nBalance at December 31,\t$ 2,870\t$ 2,871\t$ 2,355",
        "question": "What was the amount that would favorably affect the effective income tax rate if included in 2019?",
        "answer": ''
    },
    {
        "context": "Contract balances The following table provides information about receivables, contract assets and contract liabilities from contracts with customers. Accrued income relates to the Group\u2019s rights to consideration for services provided but not invoiced at the reporting date. Accrued income is transferred to receivables when invoiced. Deferred income relates to advanced consideration received for which revenue is recognised as or when services are provided. Included within deferred income is \u00a311.2m (2018: \u00a3nil) relating to consideration received from Auto Trader Auto Stock Limited (which forms part of the Group\u2019s joint venture) for the provision of data services (note 16). Revenue relating to this service is recognised on a straight-line basis over a period of 20 years.\nTable:\n\t2019\t2018\n\t\u00a3m\t\u00a3m\nReceivables, which are included in trade and other receivables\t27.0\t28.8\nAccrued income\t28.0\t26.7\nDeferred income\t(13.2)\t(1.8)",
        "question": "In which year was accrued income larger?",
        "answer": ''
    },
]
#few_shot_examples = few_shot_examples[:3]  # Use only the first 3 examples
import difflib

def is_too_similar(response, example_answer, threshold=0.8):
    similarity = difflib.SequenceMatcher(None, response, example_answer).ratio()
    return similarity > threshold
for sample in samples:
    formatted_sample = create_prompt_formats_infer(sample, few_shot_examples)

    inputs = tokenizer(
        [formatted_sample['text']],
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")

    outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=1.0,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)
    

    generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    predicted_value = generated_output.split("Answer:")[-1].strip().split("\n")[0]

    print(f"Context: {sample['context'][:100]}...")  # Print first 100 characters of context
    print(f"Question: {sample['question']}")
    print(f"Predicted Answer: {predicted_value}\n")
    print("-" * 50)  # Separator for readability


# %%
samples = [
    {
        "context": "Starbucks Corp. SBUX, +2.07% added 0.6% in thin premarket trade after the coffee chain struck a deal to push into sub-Saharan Africa next year .",
        "question": "Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of \"relation1: word1, word2; relation2: word3, word4\". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.",
        "answer": ''
    },
    {
        "context": "The company noted that the commissioning of the new cement plant is part of its Africa expansion strategy and will be the fourth in the series after Ethiopia, Zambia and Cameroun, adding that cement plants due for commissioning this year include that of Senegal and South Africa, while construction works are ongoing in several other African countries.",
        "question": "Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of \"relation1: word1, word2; relation2: word3, word4\". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.",
        "answer": ''
    },
    {
        "context": "vertex pharmaceuticals incorporated notes to consolidated financial statements ( continued ) d . stock-based compensation ( continued ) the company uses the black-scholes valuation model to estimate the fair value of stock options at the grant date . the black-scholes valuation model uses the option exercise price as well as estimates and assumptions related to the expected price volatility of the company 2019s stock , the period during which the options will be outstanding , the rate of return on risk-free investments , and the expected dividend yield for the company 2019s stock to estimate the fair value of a stock option on the grant date . the company validates its estimates and assumptions through consultations with independent third parties having relevant expertise . the fair value of each option granted under the stock and option plans during 2006 was estimated on the date of grant using the black- scholes option pricing model with the following weighted average assumptions : the weighted-average valuation assumptions were determined as follows : 2022 expected stock price volatility : in 2006 , the company changed its method of estimating expected volatility from relying exclusively on historical volatility to relying exclusively on implied volatility . options to purchase the company 2019s stock with remaining terms of greater than one year are regularly traded in the market . expected stock price volatility is calculated using the trailing one month average of daily implied volatilities prior to grant date . 2022 risk-free interest rate : the company bases the risk-free interest rate on the interest rate payable on u.s . treasury securities in effect at the time of grant for a period that is commensurate with the assumed expected option term . 2022 expected term of options : the expected term of options represents the period of time options are expected to be outstanding . the company uses historical data to estimate employee exercise and post-vest termination behavior . the company believes that all groups of employees exhibit similar exercise and post-vest termination behavior and therefore does not stratify employees into multiple groups in determining the expected term of options . 2022 expected annual dividends : the estimate for annual dividends is $ 0.00 , because the company has not historically paid , and does not intend for the foreseeable future to pay , a dividend . expected stock price volatility 57.10 % (  % ) risk-free interest rate 4.74 % (  % ) expected term 5.64 years .\n\n.\n\nTable:\n\t2006\nexpected stock price volatility\t57.10% ( 57.10 % )\nrisk-free interest rate\t4.74% ( 4.74 % )\nexpected term\t5.64 years\nexpected annual dividends\t2014",
        "question": "in 2006 what was the ratio expected stock price volatility to risk-free interest rate",
        "answer": ''
    },
    {
        "context": "item 7a . quantitative and qualitative disclosures about market risk ( amounts in millions ) in the normal course of business , we are exposed to market risks related to interest rates , foreign currency rates and certain balance sheet items . from time to time , we use derivative instruments , pursuant to established guidelines and policies , to manage some portion of these risks . derivative instruments utilized in our hedging activities are viewed as risk management tools and are not used for trading or speculative purposes . interest rates our exposure to market risk for changes in interest rates relates primarily to the fair market value and cash flows of our debt obligations . the majority of our debt ( approximately 89% ( 89 % ) and 93% ( 93 % ) as of december 31 , 2013 and 2012 , respectively ) bears interest at fixed rates . we do have debt with variable interest rates , but a 10% ( 10 % ) increase or decrease in interest rates would not be material to our interest expense or cash flows . the fair market value of our debt is sensitive to changes in interest rates , and the impact of a 10% ( 10 % ) change in interest rates is summarized below . increase/ ( decrease ) in fair market value as of december 31 , 10% ( 10 % ) increase in interest rates 10% ( 10 % ) decrease in interest rates . <table class='wikitable'><tr><td>1</td><td>as of december 31,</td><td>increase/ ( decrease ) in fair market value 10% ( 10 % ) increasein interest rates</td><td>increase/ ( decrease ) in fair market value 10% ( 10 % ) decreasein interest rates</td></tr><tr><td>2</td><td>2013</td><td>$ -26.9 ( 26.9 )</td><td>$ 27.9</td></tr><tr><td>3</td><td>2012</td><td>-27.5 ( 27.5 )</td><td>28.4</td></tr></table> we have used interest rate swaps for risk management purposes to manage our exposure to changes in interest rates . we do not have any interest rate swaps outstanding as of december 31 , 2013 . we had $ 1642.1 of cash , cash equivalents and marketable securities as of december 31 , 2013 that we generally invest in conservative , short-term bank deposits or securities . the interest income generated from these investments is subject to both domestic and foreign interest rate movements . during 2013 and 2012 , we had interest income of $ 24.7 and $ 29.5 , respectively . based on our 2013 results , a 100-basis-point increase or decrease in interest rates would affect our interest income by approximately $ 16.4 , assuming that all cash , cash equivalents and marketable securities are impacted in the same manner and balances remain constant from year-end 2013 levels . foreign currency rates we are subject to translation and transaction risks related to changes in foreign currency exchange rates . since we report revenues and expenses in u.s . dollars , changes in exchange rates may either positively or negatively affect our consolidated revenues and expenses ( as expressed in u.s . dollars ) from foreign operations . the primary foreign currencies that impacted our results during 2013 were the australian dollar , brazilian real , euro , japanese yen and the south african rand . based on 2013 exchange rates and operating results , if the u.s . dollar were to strengthen or weaken by 10% ( 10 % ) , we currently estimate operating income would decrease or increase between 3% ( 3 % ) and 4% ( 4 % ) , assuming that all currencies are impacted in the same manner and our international revenue and expenses remain constant at 2013 levels . the functional currency of our foreign operations is generally their respective local currency . assets and liabilities are translated at the exchange rates in effect at the balance sheet date , and revenues and expenses are translated at the average exchange rates during the period presented . the resulting translation adjustments are recorded as a component of accumulated other comprehensive loss , net of tax , in the stockholders 2019 equity section of our consolidated balance sheets . our foreign subsidiaries generally collect revenues and pay expenses in their functional currency , mitigating transaction risk . however , certain subsidiaries may enter into transactions in currencies other than their functional currency . assets and liabilities denominated in currencies other than the functional currency are susceptible to movements in foreign currency until final settlement . currency transaction gains or losses primarily arising from transactions in currencies other than the functional currency are included in office and general expenses . we have not entered into a material amount of foreign currency forward exchange contracts or other derivative financial instruments to hedge the effects of potential adverse fluctuations in foreign currency exchange rates. .\nQuestion: what was the interest income in 2013?\nAnswer: 24.7\nQuestion: and what was it in 2012?\nAnswer: 29.5\nQuestion: what was, then, the change in interest income over the year?\nAnswer: -4.8\nQuestion: what was the total interest income in 2012?\n",
        "question": "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided.\nAnswer the last question in the given context, if history is provided for previous questions-answers use that information (if required based on the context) to get the answer for the last question",
        "answer": ''
    },
    {
        "context": "A reconciliation of the beginning and ending balance of unrecognized tax benefits is as follows: Included in the total unrecognized tax benefits at December 31, 2019, 2018 and 2017 is $2.4 billion, $2.3 billion and $1.9 billion, respectively, that if recognized, would favorably affect the effective income tax rate.\nTable:\n\t\t\t(dollars in millions)\n\t2019\t2018\t2017\nBalance at January 1,\t$ 2,871\t$2,355\t$ 1,902\nAdditions based on tax positions related to the current year\t149\t160\t219\nAdditions for tax positions of prior years\t297\t699\t756\nReductions for tax positions of prior years\t(300)\t(248)\t(419)\nSettlements\t(58)\t(40)\t(42)\nLapses of statutes of limitations\t(89)\t(55)\t(61)\nBalance at December 31,\t$ 2,870\t$ 2,871\t$ 2,355",
        "question": "What was the amount that would favorably affect the effective income tax rate if included in 2019?",
        "answer": ''
    },
    {
        "context": "Contract balances The following table provides information about receivables, contract assets and contract liabilities from contracts with customers. Accrued income relates to the Group\u2019s rights to consideration for services provided but not invoiced at the reporting date. Accrued income is transferred to receivables when invoiced. Deferred income relates to advanced consideration received for which revenue is recognised as or when services are provided. Included within deferred income is \u00a311.2m (2018: \u00a3nil) relating to consideration received from Auto Trader Auto Stock Limited (which forms part of the Group\u2019s joint venture) for the provision of data services (note 16). Revenue relating to this service is recognised on a straight-line basis over a period of 20 years.\nTable:\n\t2019\t2018\n\t\u00a3m\t\u00a3m\nReceivables, which are included in trade and other receivables\t27.0\t28.8\nAccrued income\t28.0\t26.7\nDeferred income\t(13.2)\t(1.8)",
        "question": "In which year was accrued income larger?",
        "answer": ''
    },
]
#few_shot_examples = few_shot_examples[:3]  # Use only the first 3 examples
import difflib

def is_too_similar(response, example_answer, threshold=0.8):
    similarity = difflib.SequenceMatcher(None, response, example_answer).ratio()
    return similarity > threshold
for sample in samples:
    formatted_sample = create_prompt_formats_infer(sample, few_shot_examples)

    inputs = tokenizer(
        [formatted_sample['text']],
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")

    outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=1.0,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)
  

# %%
samples = [
    {
        "context": "Starbucks Corp. SBUX, +2.07% added 0.6% in thin premarket trade after the coffee chain struck a deal to push into sub-Saharan Africa next year .",
        "question": "Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of \"relation1: word1, word2; relation2: word3, word4\". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.",
        "answer": ''
    },
    {
        "context": "The company noted that the commissioning of the new cement plant is part of its Africa expansion strategy and will be the fourth in the series after Ethiopia, Zambia and Cameroun, adding that cement plants due for commissioning this year include that of Senegal and South Africa, while construction works are ongoing in several other African countries.",
        "question": "Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of \"relation1: word1, word2; relation2: word3, word4\". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.",
        "answer": ''
    },
    {
        "context": "vertex pharmaceuticals incorporated notes to consolidated financial statements ( continued ) d . stock-based compensation ( continued ) the company uses the black-scholes valuation model to estimate the fair value of stock options at the grant date . the black-scholes valuation model uses the option exercise price as well as estimates and assumptions related to the expected price volatility of the company 2019s stock , the period during which the options will be outstanding , the rate of return on risk-free investments , and the expected dividend yield for the company 2019s stock to estimate the fair value of a stock option on the grant date . the company validates its estimates and assumptions through consultations with independent third parties having relevant expertise . the fair value of each option granted under the stock and option plans during 2006 was estimated on the date of grant using the black- scholes option pricing model with the following weighted average assumptions : the weighted-average valuation assumptions were determined as follows : 2022 expected stock price volatility : in 2006 , the company changed its method of estimating expected volatility from relying exclusively on historical volatility to relying exclusively on implied volatility . options to purchase the company 2019s stock with remaining terms of greater than one year are regularly traded in the market . expected stock price volatility is calculated using the trailing one month average of daily implied volatilities prior to grant date . 2022 risk-free interest rate : the company bases the risk-free interest rate on the interest rate payable on u.s . treasury securities in effect at the time of grant for a period that is commensurate with the assumed expected option term . 2022 expected term of options : the expected term of options represents the period of time options are expected to be outstanding . the company uses historical data to estimate employee exercise and post-vest termination behavior . the company believes that all groups of employees exhibit similar exercise and post-vest termination behavior and therefore does not stratify employees into multiple groups in determining the expected term of options . 2022 expected annual dividends : the estimate for annual dividends is $ 0.00 , because the company has not historically paid , and does not intend for the foreseeable future to pay , a dividend . expected stock price volatility 57.10 % (  % ) risk-free interest rate 4.74 % (  % ) expected term 5.64 years .\n\n.\n\nTable:\n\t2006\nexpected stock price volatility\t57.10% ( 57.10 % )\nrisk-free interest rate\t4.74% ( 4.74 % )\nexpected term\t5.64 years\nexpected annual dividends\t2014",
        "question": "in 2006 what was the ratio expected stock price volatility to risk-free interest rate",
        "answer": ''
    },
    {
        "context": "item 7a . quantitative and qualitative disclosures about market risk ( amounts in millions ) in the normal course of business , we are exposed to market risks related to interest rates , foreign currency rates and certain balance sheet items . from time to time , we use derivative instruments , pursuant to established guidelines and policies , to manage some portion of these risks . derivative instruments utilized in our hedging activities are viewed as risk management tools and are not used for trading or speculative purposes . interest rates our exposure to market risk for changes in interest rates relates primarily to the fair market value and cash flows of our debt obligations . the majority of our debt ( approximately 89% ( 89 % ) and 93% ( 93 % ) as of december 31 , 2013 and 2012 , respectively ) bears interest at fixed rates . we do have debt with variable interest rates , but a 10% ( 10 % ) increase or decrease in interest rates would not be material to our interest expense or cash flows . the fair market value of our debt is sensitive to changes in interest rates , and the impact of a 10% ( 10 % ) change in interest rates is summarized below . increase/ ( decrease ) in fair market value as of december 31 , 10% ( 10 % ) increase in interest rates 10% ( 10 % ) decrease in interest rates . <table class='wikitable'><tr><td>1</td><td>as of december 31,</td><td>increase/ ( decrease ) in fair market value 10% ( 10 % ) increasein interest rates</td><td>increase/ ( decrease ) in fair market value 10% ( 10 % ) decreasein interest rates</td></tr><tr><td>2</td><td>2013</td><td>$ -26.9 ( 26.9 )</td><td>$ 27.9</td></tr><tr><td>3</td><td>2012</td><td>-27.5 ( 27.5 )</td><td>28.4</td></tr></table> we have used interest rate swaps for risk management purposes to manage our exposure to changes in interest rates . we do not have any interest rate swaps outstanding as of december 31 , 2013 . we had $ 1642.1 of cash , cash equivalents and marketable securities as of december 31 , 2013 that we generally invest in conservative , short-term bank deposits or securities . the interest income generated from these investments is subject to both domestic and foreign interest rate movements . during 2013 and 2012 , we had interest income of $ 24.7 and $ 29.5 , respectively . based on our 2013 results , a 100-basis-point increase or decrease in interest rates would affect our interest income by approximately $ 16.4 , assuming that all cash , cash equivalents and marketable securities are impacted in the same manner and balances remain constant from year-end 2013 levels . foreign currency rates we are subject to translation and transaction risks related to changes in foreign currency exchange rates . since we report revenues and expenses in u.s . dollars , changes in exchange rates may either positively or negatively affect our consolidated revenues and expenses ( as expressed in u.s . dollars ) from foreign operations . the primary foreign currencies that impacted our results during 2013 were the australian dollar , brazilian real , euro , japanese yen and the south african rand . based on 2013 exchange rates and operating results , if the u.s . dollar were to strengthen or weaken by 10% ( 10 % ) , we currently estimate operating income would decrease or increase between 3% ( 3 % ) and 4% ( 4 % ) , assuming that all currencies are impacted in the same manner and our international revenue and expenses remain constant at 2013 levels . the functional currency of our foreign operations is generally their respective local currency . assets and liabilities are translated at the exchange rates in effect at the balance sheet date , and revenues and expenses are translated at the average exchange rates during the period presented . the resulting translation adjustments are recorded as a component of accumulated other comprehensive loss , net of tax , in the stockholders 2019 equity section of our consolidated balance sheets . our foreign subsidiaries generally collect revenues and pay expenses in their functional currency , mitigating transaction risk . however , certain subsidiaries may enter into transactions in currencies other than their functional currency . assets and liabilities denominated in currencies other than the functional currency are susceptible to movements in foreign currency until final settlement . currency transaction gains or losses primarily arising from transactions in currencies other than the functional currency are included in office and general expenses . we have not entered into a material amount of foreign currency forward exchange contracts or other derivative financial instruments to hedge the effects of potential adverse fluctuations in foreign currency exchange rates. .\nQuestion: what was the interest income in 2013?\nAnswer: 24.7\nQuestion: and what was it in 2012?\nAnswer: 29.5\nQuestion: what was, then, the change in interest income over the year?\nAnswer: -4.8\nQuestion: what was the total interest income in 2012?\n",
        "question": "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided.\nAnswer the last question in the given context, if history is provided for previous questions-answers use that information (if required based on the context) to get the answer for the last question",
        "answer": ''
    },
    {
        "context": "A reconciliation of the beginning and ending balance of unrecognized tax benefits is as follows: Included in the total unrecognized tax benefits at December 31, 2019, 2018 and 2017 is $2.4 billion, $2.3 billion and $1.9 billion, respectively, that if recognized, would favorably affect the effective income tax rate.\nTable:\n\t\t\t(dollars in millions)\n\t2019\t2018\t2017\nBalance at January 1,\t$ 2,871\t$2,355\t$ 1,902\nAdditions based on tax positions related to the current year\t149\t160\t219\nAdditions for tax positions of prior years\t297\t699\t756\nReductions for tax positions of prior years\t(300)\t(248)\t(419)\nSettlements\t(58)\t(40)\t(42)\nLapses of statutes of limitations\t(89)\t(55)\t(61)\nBalance at December 31,\t$ 2,870\t$ 2,871\t$ 2,355",
        "question": "What was the amount that would favorably affect the effective income tax rate if included in 2019?",
        "answer": ''
    },
    {
        "context": "Contract balances The following table provides information about receivables, contract assets and contract liabilities from contracts with customers. Accrued income relates to the Group\u2019s rights to consideration for services provided but not invoiced at the reporting date. Accrued income is transferred to receivables when invoiced. Deferred income relates to advanced consideration received for which revenue is recognised as or when services are provided. Included within deferred income is \u00a311.2m (2018: \u00a3nil) relating to consideration received from Auto Trader Auto Stock Limited (which forms part of the Group\u2019s joint venture) for the provision of data services (note 16). Revenue relating to this service is recognised on a straight-line basis over a period of 20 years.\nTable:\n\t2019\t2018\n\t\u00a3m\t\u00a3m\nReceivables, which are included in trade and other receivables\t27.0\t28.8\nAccrued income\t28.0\t26.7\nDeferred income\t(13.2)\t(1.8)",
        "question": "In which year was accrued income larger?",
        "answer": ''
    },
]


# %%
import difflib

inference_template = """<s>[INST]
<<SYS>>
You are a helpful AI assistant specializing in financial analysis. Study the following examples carefully, then answer the new question using a similar approach but with information specific to the new context and question.
<</SYS>>

{user_message} [/INST]
"""

def create_prompt_formats_infer(sample, few_shot_examples):
    example_texts = []
    for i, example in enumerate(few_shot_examples[:2]):  # Limit to 2 examples
        example_text = f"Example {i+1}:\nContext: {example['context']}\nQuestion: {example['question']}\nAnswer: {example['answer']}\n\n"
        example_texts.append(example_text)
    
    current_question_text = f"Now, please answer the following NEW question:\nContext: {sample['context']}\nQuestion: {sample['question']}\nYour answer:"
    
    full_prompt = inference_template.format(user_message="".join(example_texts) + current_question_text)
    return {"text": full_prompt}

def is_too_similar(response, example_answers, threshold=0.8):
    for example_answer in example_answers:
        similarity = difflib.SequenceMatcher(None, response, example_answer).ratio()
        if similarity > threshold:
            return True
    return False

def generate_answer(prompt, max_attempts=3):
    for attempt in range(max_attempts):
        inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8 + (attempt * 0.1),  # Increase temperature with each attempt
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )
        generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        predicted_value = generated_output.split("Your answer:")[-1].strip()
        
        if attempt == max_attempts - 1 or not is_too_similar(predicted_value, [example['answer'] for example in few_shot_examples[:2]]):
            return predicted_value
        
        prompt += "\nYour previous answer was too similar to the examples. Please provide a unique answer specific to this question."
    
    return predicted_value  # Return the last generated answer if all attempts are exhausted

# Main inference loop
for sample in samples:
    formatted_sample = create_prompt_formats_infer(sample, few_shot_examples)
    
    predicted_value = generate_answer(formatted_sample['text'])

    print(f"Context: {sample['context'][:100]}...")  # Print first 100 characters of context
    print(f"Question: {sample['question']}")
    print(f"Predicted Answer: {predicted_value}\n")
    print("-" * 50)  # Separator for readability

# %%


# %%


# %%


# %%
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load your fine-tuned model
original_model = "meta-llama/Meta-Llama-3-8B"
model_path = "tatqa_l3/final_merged_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create a pipeline with max_new_tokens
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,  
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

# Create a LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# %%
from langchain.tools import BaseTool
import ast
import math

class PythonREPLTool(BaseTool):
    name = "Python REPL"
    description = "A Python REPL. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."

    def _run(self, query: str) -> str:
        try:
            # Add common math functions to the local namespace
            local_ns = {**math.__dict__}
            
            # Execute the code
            exec(query, {}, local_ns)
            
            # Capture printed output
            output = local_ns.get('_output_', '')
            return output
        except Exception as e:
            return f"Error: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("PythonREPLTool does not support async")

# %%


# %%
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory

# Initialize the Python REPL tool
python_repl = PythonREPLTool()
# Create a memory object
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the agent
agent = initialize_agent(
    [python_repl],
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

def run_agent(query):
    # Truncate the input if it's too long
    max_input_length = 1024  # Adjust this based on your model's capabilities
    tokenized_query = tokenizer.encode(query, truncation=True, max_length=max_input_length)
    truncated_query = tokenizer.decode(tokenized_query)
    
    if len(tokenized_query) == max_input_length:
        print(f"Warning: Input was truncated to {max_input_length} tokens.")
    
    return agent.run(truncated_query)

# %%
from langchain.agents import Tool
from typing import List

import operator
from typing import Dict, Any

def calculate_roi(initial_investment: float, final_value: float) -> float:
    return (final_value - initial_investment) / initial_investment * 100

def calculate_compound_interest(principal: float, rate: float, time: float, n: int = 12) -> float:
    return principal * (1 + rate/n) ** (n*time)
   

# Create tools for these functions
roi_tool = Tool(
    name="Calculate ROI",
    func=calculate_roi,
    description="Calculates the Return on Investment (ROI). Input should be two numbers: initial investment and final value."
)

compound_interest_tool = Tool(
    name="Calculate Compound Interest",
    func=calculate_compound_interest,
    description="Calculates compound interest. Input should be four numbers: principal, annual interest rate (as a decimal), time in years, and number of times interest is compounded per year (default is 12)."
)


# Add these tools to the agent
agent = initialize_agent(
    [python_repl, roi_tool, compound_interest_tool],
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# %%
prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def create_prompt_formats_infer(sample):
    instruction = sample["context"]
    input_ = sample["question"]
    output = sample["answer"]
    text = prompt_template.format(instruction, input_, output) #+ EOS_TOKEN
    return {"text": text}

samples = [
    {
        "context": "On a sequential basis, ADG revenues were up 3.3%, driven by an increase in volumes of approximately 8%, partially offset by a decrease in average selling prices of approximately 5%, mostly attributable to product mix. AMS revenues increased 12.1% driven by Analog and Imaging products. AMS increase was due to an increase of approximately 5% in average selling prices, entirely due to product mix, and to higher volumes of approximately of 7%. MDG revenues increased by 7.9%, mainly driven by Microcontrollers, due to both higher average selling prices of approximately 6%, entirely due to product mix, and higher volumes of approximately 2%. On a year-over-year basis, fourth quarter net revenues increased by 4.0%. ADG revenues decreased 4.5% compared to the year-ago quarter on lower revenues in both Automotive and Power Discrete. The decrease was entirely due to lower average selling prices of approximately 4%, while volumes remained substantially flat. The decrease in average selling prices was a combination of less favorable product mix and lower selling prices. AMS fourth quarter revenues grew 9.9% year-over-year, mainly driven by Analog and Imaging. The increase was entirely due to higher average selling prices of approximately 18%, entirely attributable to product mix,\npartially offset by lower volumes of approximately 8%. MDG fourth quarter revenues increased by 7.6%, mainly driven by Microcontrollers. The increase was due to higher average selling prices of approximately 9%,\nentirely due to improved product mix.\nTable:\n\t\tThree Months Ended\t\t% Variation\t\n\tDecember 31, 2019\tSeptember 29, 2019\tDecember 31, 2018\tSequential\tYear-Over-Year\n\t\t\t(Unaudited, in millions)\t\t\nAutomotive and Discrete Group (ADG)\t$924\t$894\t$967\t3.3%\t(4.5)%\nAnalog, MEMS and Sensors Group (AMS)\t1,085\t968\t988\t12.1\t9.9\nMicrocontrollers and Digital ICs Group (MDG)\t742\t688\t689\t7.9\t7.6\nOthers\t3\t3\t4\t\u2014\t\u2014\nTotal consolidated net revenues\t$2,754\t$2,553\t$2,648\t7.9%\t4.0%",
        "question": "What is the average net revenues from Analog, MEMS and Sensors Group (AMS) for the period December 31, 2019 and 2018?",
        "answer": '' #"1036.5"
    },
    {
        "context": "devon energy corporation and subsidiaries notes to consolidated financial statements 2014 ( continued ) the following methods and assumptions were used to estimate the fair values in the tables above . fixed-income securities 2014 devon 2019s fixed-income securities consist of u.s . treasury obligations , bonds issued by investment-grade companies from diverse industries , and asset-backed securities . these fixed-income securities are actively traded securities that can be redeemed upon demand . the fair values of these level 1 securities are based upon quoted market prices . devon 2019s fixed income securities also include commingled funds that primarily invest in long-term bonds and u.s . treasury securities . these fixed income securities can be redeemed on demand but are not actively traded . the fair values of these level 2 securities are based upon the net asset values provided by the investment managers . equity securities 2014 devon 2019s equity securities include a commingled global equity fund that invests in large , mid and small capitalization stocks across the world 2019s developed and emerging markets . these equity securities can be redeemed on demand but are not actively traded . the fair values of these level 2 securities are based upon the net asset values provided by the investment managers . at december 31 , 2010 , devon 2019s equity securities consisted of investments in u.s . large and small capitalization companies and international large capitalization companies . these equity securities were actively traded securities that could be redeemed upon demand . the fair values of these level 1 securities are based upon quoted market prices . at december 31 , 2010 , devon 2019s equity securities also included a commingled fund that invested in large capitalization companies . these equity securities could be redeemed on demand but were not actively traded . the fair values of these level 2 securities are based upon the net asset values provided by the investment managers . other securities 2014 devon 2019s other securities include commingled , short-term investment funds . these securities can be redeemed on demand but are not actively traded . the fair values of these level 2 securities are based upon the net asset values provided by investment managers . devon 2019s hedge fund and alternative investments include an investment in an actively traded global mutual fund that focuses on alternative investment strategies and a hedge fund of funds that invests both long and short using a variety of investment strategies . devon 2019s hedge fund of funds is not actively traded and devon is subject to redemption restrictions with regards to this investment . the fair value of this level 3 investment represents the fair value as determined by the hedge fund manager . included below is a summary of the changes in devon 2019s level 3 plan assets ( in millions ) . .\n\n.\n\nTable:\ndecember 31 2009\t$ 51\npurchases\t3\ninvestment returns\t4\ndecember 31 2010\t58\npurchases\t33\ninvestment returns\t-1 ( 1 )\ndecember 31 2011\t$ 90",
        "question": "what was the percentage change in devon 2019s level 3 plan assets from 2009 to 2010",
        "answer": '' #"13.7%"
    },
]
for sample in samples:
    formatted_sample = create_prompt_formats_infer(sample)
    result = run_agent(formatted_sample['text'])
    print(result)

# %%


# %%
from langchain.llms.base import LLM
from typing import Any, List, Optional
import torch

class FineTunedLLM(LLM):
    model: Any
    tokenizer: Any

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer([prompt], return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=64, use_cache=True)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    @property
    def _identifying_params(self) -> dict:
        return {"name_of_model": "Tatqa LLaMA3"}

    @property
    def _llm_type(self) -> str:
        return "custom"

# %%
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import operator
import re

def process_model_output(output: str, calculator_func):
    # Check if there's a calculation needed
    calculation_match = re.search(r'Calculate.*?(\d+\.?\d*\s*[\+\-\*/]\s*\d+\.?\d*)', output)
    if calculation_match:
        calculation = calculation_match.group(1)
        result = calculator_func(calculation)
        # Replace the calculation request with the result
        output = output.replace(calculation_match.group(0), f"Calculation result: {result}")
    
    return output

def calculator(expression: str) -> str:
    try:
        parts = expression.split()
        if len(parts) != 3:
            return "Error: Expression should be in the format 'number operator number'"
        
        num1, op, num2 = parts
        num1, num2 = float(num1), float(num2)
        
        allowed_operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else "Error: Division by zero",
        }
        
        if op not in allowed_operations:
            return f"Error: Unsupported operator '{op}'. Allowed operators are: {', '.join(allowed_operations.keys())}"
        
        result = allowed_operations[op](num1, num2)
        return str(result)
    except ValueError:
        return "Error: Invalid number format"
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize your custom LLM
llm = FineTunedLLM(model, tokenizer)

# Define the tools list
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for when you need to perform mathematical calculations. The input should be a simple mathematical expression with two numbers and one operator, e.g., '23 + 7' or '15 * 3'."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

# %%
samples = [
    {
        "context": "FINANCIAL ASSETS AT FAIR VALUE THROUGH PROFIT OR LOSS (continued) Movement of FVPL is analysed as follows: Note: During the year ended 31 December 2019, the Group\u2019s additions to FVPL mainly comprised the following: an investment in a retail company of approximately USD500 million (equivalent to approximately RMB3,550 million) to subscribe for approximately 21% of its equity interests in form of preferred shares, on an outstanding basis; an additional investment in a real estate O2O platform in the PRC of approximately USD320 million (equivalent to approximately RMB2,258 million). As at 31 December 2019, the Group\u2019s equity interests in this investee company are approximately 9% on an outstanding basis; and new investments and additional investments with an aggregate amount of approximately RMB38,810 million in listed and unlisted entities mainly operating in the United States, the PRC and other Asian countries. These companies are principally engaged in social networks, Internet platform, technology and other Internet-related business. None of the above investment was individually significant that triggers any disclosure requirements pursuant to Chapter 14 of the Listing Rules at the time of inception. During the year ended 31 December 2019, except as described in Note 21(b), transfers also mainly comprised an equity investment designated as FVOCI due to the conversion of the redeemable instruments into ordinary shares amounting to RMB1,395 million upon its IPO. During the year ended 31 December 2019, the Group disposed of certain investments with an aggregate amount of RMB16,664 million, which are mainly engaged in the provision of Internet-related services. Management has assessed the level of influence that the Group exercises on certain FVPL with shareholding exceeding 20%. Since these investments are either held in form of redeemable instruments or interests in limited life partnership without significant influence, these investments have been classified as FVPL.\nTable:\n\t2019\t2018\n\tRMB\u2019Million\tRMB\u2019Million\nAt beginning of the year\t97,877\t\u2013\nAdjustment on adoption of IFRS 9\t\u2013\t95,497\nAdditions (Note (a))\t44,618\t60,807\nTransfers (Note (b))\t(1,421)\t(78,816)\nChanges in fair value (Note 7(b))\t9,511\t28,738\nDisposals (Note (c))\t(16,664)\t(14,805)\nCurrency translation differences\t2,015\t6,456\nAt end of the year\t135,936\t97,877",
        "question": "How much is the change in disposals between 2018 year end and 2019 year end?",
        "answer": '' #"-1859"
    },
    {
        "context": "a. Analysis of revenue growth On a reported basis, TCS\u2019 revenue grew 19% in FY 2019, compared to 4.4% in the prior year. This was largely an outcome of greater demand for our services and solutions during the year, driven by expanding participation in our customers\u2019 growth and transformation initiatives. In addition, there was some benefit from the movement in currency exchange rates. FY 2019 saw volatility in USD-INR, ranging from `64.90 and `74.10, and averaging at `70.07. There was also significant volatility in exchange rates of emerging markets\u2019 currencies. Average currency exchange rates during FY 2019 for the three major currencies are given below: Movements in currency exchange rates through the year resulted in a positive impact of 7.6% on the reported revenue. The constant currency revenue growth for the year, which is the reported revenue growth stripped of the currency impact, was 11.4%.\nTable:\nCurrency\tWeightage (%)\tFY 2019 `\tFY 2018 `\t% Change YoY\nUSD\t53.6\t70.07\t64.49\t8.7\nGBP\t13.9\t91.60\t86.05\t6.5\nEUR\t10.1\t80.82\t76.16\t6.1\nBreakup of revenue growth\tFY 2019 (%)\tFY 2018 (%)\t\t\nBusiness growth\t11.4\t6.7\t\t\nImpact of exchange rate\t7.6\t(2.3)\t\t\nTotal growth\t19.0\t4.4\t\t",
        "question": "What was the average USD-INR exchange rate in FY 2019?",
        "answer": '' #"`70.07"
    },
]
for sample in samples:
    formatted_sample = create_prompt_formats_infer(sample)

    # Tokenize the formatted prompt
    inputs = tokenizer(
        [formatted_sample['text']],
        return_tensors="pt"
    )

    # Generate the output
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

    # Decode the output
    generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Process the output and perform calculations if needed
    processed_output = process_model_output(generated_output, calculator)

    print(f"Processed Output: {processed_output}")

# %%
import operator

class ArithmeticTool:
    def __init__(self):
        self.operations = {
            'add': operator.add,
            'subtract': operator.sub,
            'multiply': operator.mul,
            'divide': operator.truediv,
            'percentage': lambda x, y: (y - x) / x * 100 if x != 0 else float('inf')
        }

    def compute(self, operation, *args):
        if operation not in self.operations:
            return f"Error: Unknown operation '{operation}'"
        
        try:
            args = [float(arg) for arg in args]
            result = self.operations[operation](*args)
            return round(result, 4)
        except ValueError:
            return "Error: Invalid numeric input"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"

arithmetic_tool = ArithmeticTool()

# %%
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the fine-tuned model
model_path = "3006_final_llama3/final_merged_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# %%

# Create a pipeline with max_new_tokens
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.5,
    top_p=0.95,
    repetition_penalty=1.15
)

# Create a LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create a prompt template
template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Use the tool to calculate if required

### Instruction: {context}

### Input: {question}

### Response:  

Example: 

### Instruction: Actuarial assumptions The Group\u2019s scheme liabilities are measured using the projected unit credit method using the principal actuarial assumptions set out below: Notes: 1 Figures shown represent a weighted average assumption of the individual schemes. 2 The rate of increases in pensions in payment and deferred revaluation are dependent on the rate of inflation.\nTable:\n\t2019 %\t2018 %\t2017 %\nWeighted average actuarial assumptions used at 31 March1:\t\t\t\nRate of inflation2\t2.9\t2.9\t3.0\nRate of increase in salaries\t2.7\t2.7\t2.6\nDiscount rate\t2.3\t2.5\t2.6

### Input: How much is the 2019 rate of inflation?

### Response: 2.9

"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Create an LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# %%
import re

def process_output(output):
    def replace_calculation(match):
        parts = match.group(1).split()
        operation = parts[0]
        args = parts[1:]
        result = arithmetic_tool.compute(operation, *args)
        return str(result)

    # Replace all [CALCULATE ...] blocks with their results
    processed_output = re.sub(r'\[CALCULATE (.*?)\]', replace_calculation, output)
    return processed_output

def run_chain(context, question):
    # Run the LLMChain
    result = chain.run(context=context, question=question)
    
    # Process the output to perform calculations
    processed_result = process_output(result)
    final_response = processed_result.split("### Response:")[-1].strip()
    return final_response

# %%
context = "FINANCIAL ASSETS AT FAIR VALUE THROUGH PROFIT OR LOSS (continued) Movement of FVPL is analysed as follows: Note: During the year ended 31 December 2019, the Group\u2019s additions to FVPL mainly comprised the following: an investment in a retail company of approximately USD500 million (equivalent to approximately RMB3,550 million) to subscribe for approximately 21% of its equity interests in form of preferred shares, on an outstanding basis; an additional investment in a real estate O2O platform in the PRC of approximately USD320 million (equivalent to approximately RMB2,258 million). As at 31 December 2019, the Group\u2019s equity interests in this investee company are approximately 9% on an outstanding basis; and new investments and additional investments with an aggregate amount of approximately RMB38,810 million in listed and unlisted entities mainly operating in the United States, the PRC and other Asian countries. These companies are principally engaged in social networks, Internet platform, technology and other Internet-related business. None of the above investment was individually significant that triggers any disclosure requirements pursuant to Chapter 14 of the Listing Rules at the time of inception. During the year ended 31 December 2019, except as described in Note 21(b), transfers also mainly comprised an equity investment designated as FVOCI due to the conversion of the redeemable instruments into ordinary shares amounting to RMB1,395 million upon its IPO. During the year ended 31 December 2019, the Group disposed of certain investments with an aggregate amount of RMB16,664 million, which are mainly engaged in the provision of Internet-related services. Management has assessed the level of influence that the Group exercises on certain FVPL with shareholding exceeding 20%. Since these investments are either held in form of redeemable instruments or interests in limited life partnership without significant influence, these investments have been classified as FVPL.\nTable:\n\t2019\t2018\n\tRMB\u2019Million\tRMB\u2019Million\nAt beginning of the year\t97,877\t\u2013\nAdjustment on adoption of IFRS 9\t\u2013\t95,497\nAdditions (Note (a))\t44,618\t60,807\nTransfers (Note (b))\t(1,421)\t(78,816)\nChanges in fair value (Note 7(b))\t9,511\t28,738\nDisposals (Note (c))\t(16,664)\t(14,805)\nCurrency translation differences\t2,015\t6,456\nAt end of the year\t135,936\t97,877"

question = "How much is the change in disposals between 2018 year end and 2019 year end?"

result = run_chain(context, question)
# print(f"Context: {context[:100]}...") # Print first 100 characters of context
# print(f"Question: {question}")
print(f"Predicted Answer: {result}")
print("-" * 50) # Separator for readability

# %%


# %%


# %%
def calculator(expression: str) -> str:
    try:
        parts = expression.split()
        if len(parts) != 3:
            return "Error: Expression should be in the format 'number operator number'"
        
        num1, op, num2 = parts
        num1, num2 = float(num1), float(num2)
        
        allowed_operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else "Error: Division by zero",
        }
        
        if op not in allowed_operations:
            return f"Error: Unsupported operator '{op}'. Allowed operators are: {', '.join(allowed_operations.keys())}"
        
        result = allowed_operations[op](num1, num2)
        return str(result)
    except ValueError:
        return "Error: Invalid number format"
    except Exception as e:
        return f"Error: {str(e)}"

# %%
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.llms.base import LLM
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.chains import LLMChain
from typing import List, Union, Optional, Dict, Any
import re

# Custom LLM class
class FineTunedLLM(LLM):
    model: Any
    tokenizer: Any

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=64, use_cache=True)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"name_of_model": "Fine-tuned LLaMA"}

    @property
    def _llm_type(self) -> str:
        return "custom"

# Custom output parser
class CustomOutputParser:
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if there's a calculation needed
        calculation_match = re.search(r'Calculate.*?(\d+\.?\d*\s*[\+\-\*/]\s*\d+\.?\d*)', llm_output)
        if calculation_match:
            action = "Calculator"
            action_input = calculation_match.group(1)
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
        
        # If no calculation is needed, assume it's the final answer
        return AgentFinish(return_values={"output": llm_output}, log=llm_output)

# Custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        
        kwargs["thoughts"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# Calculator function
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "Error: Invalid expression"

# Load the model and tokenizer
model_name = "tatqa_l3/final_merged_checkpoint"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the custom LLM
llm = FineTunedLLM(model, tokenizer)

# Set up the calculator tool
calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Useful for when you need to perform mathematical calculations."
)

# Define the prompt template
prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Context: the context information provided
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

{thoughts}

Context: {context}
Question: {question}

{agent_scratchpad}
"""

# Create the custom prompt
prompt = CustomPromptTemplate(
    template=prompt_template,
    tools=[calculator_tool],
    input_variables=["context", "question", "intermediate_steps", "agent_scratchpad"]
)

# Create the output parser
output_parser = CustomOutputParser()

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Create the agent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=CustomOutputParser(),  # Use an instance of CustomOutputParser
    stop=["\nObservation:"],
    allowed_tools=["Calculator"]
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=[calculator_tool], verbose=True)

# Define the samples
samples = [
    {
        "context": "FINANCIAL ASSETS AT FAIR VALUE THROUGH PROFIT OR LOSS (continued) Movement of FVPL is analysed as follows: Note: During the year ended 31 December 2019, the Group\u2019s additions to FVPL mainly comprised the following: an investment in a retail company of approximately USD500 million (equivalent to approximately RMB3,550 million) to subscribe for approximately 21% of its equity interests in form of preferred shares, on an outstanding basis; an additional investment in a real estate O2O platform in the PRC of approximately USD320 million (equivalent to approximately RMB2,258 million). As at 31 December 2019, the Group\u2019s equity interests in this investee company are approximately 9% on an outstanding basis; and new investments and additional investments with an aggregate amount of approximately RMB38,810 million in listed and unlisted entities mainly operating in the United States, the PRC and other Asian countries. These companies are principally engaged in social networks, Internet platform, technology and other Internet-related business. None of the above investment was individually significant that triggers any disclosure requirements pursuant to Chapter 14 of the Listing Rules at the time of inception. During the year ended 31 December 2019, except as described in Note 21(b), transfers also mainly comprised an equity investment designated as FVOCI due to the conversion of the redeemable instruments into ordinary shares amounting to RMB1,395 million upon its IPO. During the year ended 31 December 2019, the Group disposed of certain investments with an aggregate amount of RMB16,664 million, which are mainly engaged in the provision of Internet-related services. Management has assessed the level of influence that the Group exercises on certain FVPL with shareholding exceeding 20%. Since these investments are either held in form of redeemable instruments or interests in limited life partnership without significant influence, these investments have been classified as FVPL.\nTable:\n\t2019\t2018\n\tRMB\u2019Million\tRMB\u2019Million\nAt beginning of the year\t97,877\t\u2013\nAdjustment on adoption of IFRS 9\t\u2013\t95,497\nAdditions (Note (a))\t44,618\t60,807\nTransfers (Note (b))\t(1,421)\t(78,816)\nChanges in fair value (Note 7(b))\t9,511\t28,738\nDisposals (Note (c))\t(16,664)\t(14,805)\nCurrency translation differences\t2,015\t6,456\nAt end of the year\t135,936\t97,877",
        "question": "How much is the change in disposals between 2018 year end and 2019 year end?",
        "answer": '' #"-1859"
    },
    {
        "context": "a. Analysis of revenue growth On a reported basis, TCS\u2019 revenue grew 19% in FY 2019, compared to 4.4% in the prior year. This was largely an outcome of greater demand for our services and solutions during the year, driven by expanding participation in our customers\u2019 growth and transformation initiatives. In addition, there was some benefit from the movement in currency exchange rates. FY 2019 saw volatility in USD-INR, ranging from `64.90 and `74.10, and averaging at `70.07. There was also significant volatility in exchange rates of emerging markets\u2019 currencies. Average currency exchange rates during FY 2019 for the three major currencies are given below: Movements in currency exchange rates through the year resulted in a positive impact of 7.6% on the reported revenue. The constant currency revenue growth for the year, which is the reported revenue growth stripped of the currency impact, was 11.4%.\nTable:\nCurrency\tWeightage (%)\tFY 2019 `\tFY 2018 `\t% Change YoY\nUSD\t53.6\t70.07\t64.49\t8.7\nGBP\t13.9\t91.60\t86.05\t6.5\nEUR\t10.1\t80.82\t76.16\t6.1\nBreakup of revenue growth\tFY 2019 (%)\tFY 2018 (%)\t\t\nBusiness growth\t11.4\t6.7\t\t\nImpact of exchange rate\t7.6\t(2.3)\t\t\nTotal growth\t19.0\t4.4\t\t",
        "question": "What was the average USD-INR exchange rate in FY 2019?",
        "answer": '' #"`70.07"
    },
]

# Process samples
for sample in samples:
    context = sample['context']
    question = sample['question']
    
    # Use the agent to generate a response
    response = agent_executor.run(context=context, question=question)
    
    print(f"Question: {question}")
    print(f"Agent Response: {response}")
    print("\n---\n")


# %%
samples = [
    {
        "context": "FINANCIAL ASSETS AT FAIR VALUE THROUGH PROFIT OR LOSS (continued) Movement of FVPL is analysed as follows: Note: During the year ended 31 December 2019, the Group\u2019s additions to FVPL mainly comprised the following: an investment in a retail company of approximately USD500 million (equivalent to approximately RMB3,550 million) to subscribe for approximately 21% of its equity interests in form of preferred shares, on an outstanding basis; an additional investment in a real estate O2O platform in the PRC of approximately USD320 million (equivalent to approximately RMB2,258 million). As at 31 December 2019, the Group\u2019s equity interests in this investee company are approximately 9% on an outstanding basis; and new investments and additional investments with an aggregate amount of approximately RMB38,810 million in listed and unlisted entities mainly operating in the United States, the PRC and other Asian countries. These companies are principally engaged in social networks, Internet platform, technology and other Internet-related business. None of the above investment was individually significant that triggers any disclosure requirements pursuant to Chapter 14 of the Listing Rules at the time of inception. During the year ended 31 December 2019, except as described in Note 21(b), transfers also mainly comprised an equity investment designated as FVOCI due to the conversion of the redeemable instruments into ordinary shares amounting to RMB1,395 million upon its IPO. During the year ended 31 December 2019, the Group disposed of certain investments with an aggregate amount of RMB16,664 million, which are mainly engaged in the provision of Internet-related services. Management has assessed the level of influence that the Group exercises on certain FVPL with shareholding exceeding 20%. Since these investments are either held in form of redeemable instruments or interests in limited life partnership without significant influence, these investments have been classified as FVPL.\nTable:\n\t2019\t2018\n\tRMB\u2019Million\tRMB\u2019Million\nAt beginning of the year\t97,877\t\u2013\nAdjustment on adoption of IFRS 9\t\u2013\t95,497\nAdditions (Note (a))\t44,618\t60,807\nTransfers (Note (b))\t(1,421)\t(78,816)\nChanges in fair value (Note 7(b))\t9,511\t28,738\nDisposals (Note (c))\t(16,664)\t(14,805)\nCurrency translation differences\t2,015\t6,456\nAt end of the year\t135,936\t97,877",
        "question": "How much is the change in disposals between 2018 year end and 2019 year end?",
        "answer": '' #"-1859"
    },
    {
        "context": "a. Analysis of revenue growth On a reported basis, TCS\u2019 revenue grew 19% in FY 2019, compared to 4.4% in the prior year. This was largely an outcome of greater demand for our services and solutions during the year, driven by expanding participation in our customers\u2019 growth and transformation initiatives. In addition, there was some benefit from the movement in currency exchange rates. FY 2019 saw volatility in USD-INR, ranging from `64.90 and `74.10, and averaging at `70.07. There was also significant volatility in exchange rates of emerging markets\u2019 currencies. Average currency exchange rates during FY 2019 for the three major currencies are given below: Movements in currency exchange rates through the year resulted in a positive impact of 7.6% on the reported revenue. The constant currency revenue growth for the year, which is the reported revenue growth stripped of the currency impact, was 11.4%.\nTable:\nCurrency\tWeightage (%)\tFY 2019 `\tFY 2018 `\t% Change YoY\nUSD\t53.6\t70.07\t64.49\t8.7\nGBP\t13.9\t91.60\t86.05\t6.5\nEUR\t10.1\t80.82\t76.16\t6.1\nBreakup of revenue growth\tFY 2019 (%)\tFY 2018 (%)\t\t\nBusiness growth\t11.4\t6.7\t\t\nImpact of exchange rate\t7.6\t(2.3)\t\t\nTotal growth\t19.0\t4.4\t\t",
        "question": "What was the average USD-INR exchange rate in FY 2019?",
        "answer": '' #"`70.07"
    },
]
# Process samples
for sample in samples:
    context = sample['context']
    question = sample['question']
    
    # Use the agent to generate a response
    response = agent_executor.run(context=context, question=question)
    
    print(f"Question: {question}")
    print(f"Agent Response: {response}")
    print("\n---\n")


