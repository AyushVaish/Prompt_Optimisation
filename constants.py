
PROMPT_ENGINEERING_BEST_PRACTICES = [
    "1. Use the latest model. For best results, we generally recommend using the latest, most capable models. Newer models tend to be easier to prompt engineer.",
    "2. Put instructions at the beginning of the prompt and use ### or \"\"\" to separate the instruction and context. \n"
    "   • Less effective ❌:\n"
    "     Summarize the text below as a bullet point list...\n"
    "     {text input here}\n"
    "   • Better ✅:\n"
    "     Summarize the text below as a bullet point list...\n"
    "     Text: \"\"\"\n"
    "     {text input here}\n"
    "     \"\"\"",
    "3. Be specific, descriptive, and as detailed as possible about the desired context, outcome, length, format, style, etc. \n"
    "   • Less effective ❌: Write a poem about OpenAI.\n"
    "   • Better ✅: Write a short inspiring poem about OpenAI, focusing on the recent DALL-E product launch in the style of a {famous poet}.",
    "4. Articulate the desired output format through examples. Show and tell — the model responds better when shown specific format requirements. \n"
    "   • Less effective ❌: Extract the entities mentioned in the text below...\n"
    "   • Better ✅: Here is a JSON example of the format I’d like:\n"
    "     ```json\n"
    "     [\n"
    "       {\n"
    "         \"Company Name\": \"OpenAI\",\n"
    "         \"Person Name\": \"Sam Altman\",\n"
    "         \"Topics\": [\"GPT-4\", \"ChatGPT\"],\n"
    "         \"Themes\": [\"AI safety\", \"ML research\"]\n"
    "       },\n"
    "       ...\n"
    "     ]\n"
    "     ```\n"
    "     Now, extract similar entities from this text."
]
