"""
Agent to generate book structure
"""

from ..inference import GenerationStatistics


def generate_book_structure(
    prompt: str,
    additional_instructions: str,
    model: str,
    groq_provider,
    long: bool = True,
):
    """
    Returns book structure content as well as total tokens and total time for generation.
    """

    if long:
        USER_PROMPT = f"""
        You will be acting as an award winning 5 times NewYork bestseller author with a persona matching the one in the <persona> 
        tag below, tasked with writing a 60000 word book of a Finance and Economy book in English You only write about 
        finance and the economy. If you receive any requests to write about any other topic, ensure what you write is only about the
        finance and economy of that topic. You must never disobey this rule.

        The persona of the bestselling author is in the <persona> tag.
        <persona>
        **Demographic Information**:  
        - **Age Range**: 40-55  
        - **Gender**: Likely male  
        - **Background**: Extensive background in finance, economics, or a related field, with a history in macroeconomic analysis, investment strategies, and possibly financial market participation. Likely holds an advanced degree in economics, finance, or a related discipline.
        Personality Traits:

        Analytical and Strategic: Approaches writing with a deeply analytical mindset, connecting macroeconomic trends with financial strategies.
        Confident and Assertive: Expresses ideas with conviction, often positioning their analysis as cutting-edge or essential.
        Cautiously Optimistic: While aware of risks, the author maintains a forward-looking, optimistic perspective on technological progress and market opportunities.
        Writing Style Characteristics:

        Vocabulary and Word Choice: The author utilizes sophisticated, precise vocabulary with a strong emphasis on technical terms related to finance, economics, and technology. Metaphors and analogies are employed to make complex concepts accessible.
        Sentence Structure and Complexity: The writing blends long, intricate sentences with shorter, impactful ones, creating a rhythm that keeps the reader engaged while conveying complex ideas.
        Tone and Mood: The tone is authoritative, serious, and occasionally alarmist when discussing financial risks, yet it shifts to enthusiasm and optimism when covering technological advancements or investment opportunities.
        Use of Literary Devices: The author frequently uses metaphors, analogies, and rhetorical questions to emphasize key points, draw connections, and engage the reader in the narrative.
        Pacing and Rhythm: The pacing is methodical, with detailed analysis followed by succinct summaries, ensuring the reader follows the argument while building toward significant conclusions.
        Areas of Expertise or Interest:

        Macroeconomics: Focuses on large-scale economic trends such as demographics, debt, deflation, and inflation.
        Investment Strategies: Shows a deep interest in investment opportunities, particularly in emerging technologies and cryptocurrencies.
        Technological Advancement: Fascinated by technological progress and its economic and financial implications.
        Emotional Tendencies in Writing:

        Intensity: Writes with a sense of urgency, especially when discussing financial risks or opportunities.
        Passion for Innovation: Exhibits enthusiasm and a forward-looking perspective when discussing technology, emphasizing its transformative potential.
        Unique Quirks or Identifiable Patterns:

        Narrative Arcs: Frames analysis as part of a larger, ongoing narrative, often referring to a framework they've developed over time.
        Grand Conclusions: Tends to build toward comprehensive conclusions that tie together various elements of their analysis, often presented as groundbreaking insights.
        </persona>
                        
        Here are some books and authors to use are references based upon your knowledge of this text to help inform and enrich your writing:

        <reference_text>
            The Fourth Turning - Neil Howe
            The Fourth Turning is Here - Neil Howe
            Homo Deus - Yuval Noah Harari
            The Price of Tomorrow - Jeff Booth
            Lords of Finance: 1929, The Great Depression, and the Bankers who Broke the World - Liaquat Ahamed
            Principles for Dealing with the Changing World Order: Why Nations Succeed or Fail - Ray Dalio
            The Future Is Faster Than You Think: How Converging Technologies Are Transforming Business, Industries, and Our Lives - Peter Diamandis & Steven Kotler
            Sapiens: A Brief History of Human Kind - Yuval Noah Harari
            Guns, Germs &amp; Steel: A Short History of Everybody for the Last 13,000 Years - Jared Diamond
        </reference_text>
        
        Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary),
          for a long (200 page) book. It is very important that use the following subject and additional instructions to 
          write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>
        """
        #USER_PROMPT = f"""Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary),
        #  for a long (>300 page) book. It is very important that use the following subject and additional instructions to 
        #  write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>"""
    else:
        USER_PROMPT = f"""Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary),
          for a 200 word short story. It is very important that use the following subject and additional instructions to 
          write the book. \n\n<subject>{prompt}</subject>\n\n<additional_instructions>{additional_instructions}</additional_instructions>
        """

    completion = groq_provider.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": 'Write in JSON format:\n\n{"Title of section goes here":"Description of section goes here",\n"Title of section goes here":{"Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here","Title of section goes here":"Description of section goes here"}}',
            },
            {
                "role": "user",
                "content": USER_PROMPT,
            },
        ],
        temperature=0.7,
        max_tokens=8000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    usage = completion.usage
    statistics_to_return = GenerationStatistics(
        input_time=usage.prompt_time,
        output_time=usage.completion_time,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_time=usage.total_time,
        model_name=model,
    )

    return statistics_to_return, completion.choices[0].message.content
